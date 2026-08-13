"""Tests for src/integrations/llm/openai_utils.py."""

import logging
from unittest.mock import MagicMock, Mock, call, patch
import pytest

from src.integrations.llm import openai_utils


# ============================================================================
# 1. Client Builder Tests
# ============================================================================

class TestBuildOpenAIClient:
    """Tests for build_openai_client."""

    def test_missing_openai_dependency_raises_runtime_error(self, monkeypatch):
        """Test ImportError during openai import raises RuntimeError."""
        # Arrange
        monkeypatch.setattr("builtins.__import__", Mock(side_effect=ImportError("No module named openai")))

        # Act & Assert
        with pytest.raises(RuntimeError, match="Missing dependency 'openai'"):
            openai_utils.build_openai_client(api_key="test-key", timeout_sec=10.0)

    @pytest.mark.parametrize("invalid_key", ["", "   ", "\t\n"])
    def test_empty_api_key_raises_runtime_error(self, invalid_key):
        """Test empty or whitespace-only API keys raise RuntimeError."""
        # Act & Assert
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing"):
            openai_utils.build_openai_client(api_key=invalid_key, timeout_sec=10.0)

    def test_build_client_with_timeout_success(self):
        """Test building client with timeout parameter."""
        # Arrange
        mock_openai_cls = MagicMock(return_value="mock_client_instance")

        # Act
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            client = openai_utils.build_openai_client(api_key="valid-key", timeout_sec=15.0)

        # Assert
        assert client == "mock_client_instance"
        mock_openai_cls.assert_called_once_with(api_key="valid-key", timeout=15.0)

    def test_build_client_fallback_without_timeout(self):
        """Test fallback when OpenAI constructor rejects timeout argument (TypeError)."""
        # Arrange
        mock_openai_cls = MagicMock(
            side_effect=[TypeError("unexpected keyword argument 'timeout'"), "mock_client_fallback"]
        )

        # Act
        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            client = openai_utils.build_openai_client(api_key="valid-key", timeout_sec=15.0)

        # Assert
        assert client == "mock_client_fallback"
        assert mock_openai_cls.call_count == 2
        mock_openai_cls.assert_called_with(api_key="valid-key")


# ============================================================================
# 2. HTTP Status & Retry-After Extractor Tests
# ============================================================================

class TestExtractors:
    """Tests status code and retry-after header extraction from exceptions."""

    @pytest.mark.parametrize(
        ("exc_builder", "expected_status"),
        [
            (lambda: setattr(Exception(), "status_code", 404) or Exception(), None),  # Direct attr handled below
            (lambda: type("E", (Exception,), {"status_code": 404})(), 404),
            (lambda: type("E", (Exception,), {"response": Mock(status_code=503)})(), 503),
            (lambda: Exception(), None),
            (lambda: type("E", (Exception,), {"status_code": "404"})(), None),
            (lambda: type("E", (Exception,), {"response": Mock(status_code=None)})(), None),
        ],
    )
    def test_extract_http_status_code(self, exc_builder, expected_status):
        """Test status code extraction across direct, nested response, and missing formats."""
        # Arrange
        exc = exc_builder()

        # Act
        status = openai_utils.extract_http_status_code(exc)

        # Assert
        assert status == expected_status

    @pytest.mark.parametrize(
        ("exc_builder", "expected_delay"),
        [
            (lambda: Exception(), None),
            (lambda: type("E", (Exception,), {"response": Mock(spec=[], headers=None)})(), None),
            (lambda: type("E", (Exception,), {"response": object()})(), None),
            (lambda: type("E", (Exception,), {"response": Mock(headers={"retry-after": "5.5"})})(), 5.5),
            (lambda: type("E", (Exception,), {"response": Mock(headers={"Retry-After": "10"})})(), 10.0),
            (lambda: type("E", (Exception,), {"response": Mock(headers={"retry-after": "invalid-float"})})(), None),
            (lambda: type("E", (Exception,), {"response": Mock(headers={"retry-after": "-3.5"})})(), 0.0),
        ],
    )
    def test_extract_retry_after_seconds(self, exc_builder, expected_delay):
        """Test retry-after extraction across header formats, non-numeric values, and missing attributes."""
        # Arrange
        exc = exc_builder()

        # Act
        delay = openai_utils.extract_retry_after_seconds(exc)

        # Assert
        assert delay == expected_delay


# ============================================================================
# 3. Error Classification Tests
# ============================================================================

class TestClassifyOpenAIError:
    """Tests classification of exceptions into (retryable, retry_after, reason)."""

    @pytest.mark.parametrize(
        ("status_code", "retry_after_header", "expected"),
        [
            (429, "3.0", (True, 3.0, "http_429")),
            (429, None, (True, None, "http_429")),
            (408, None, (True, None, "http_408")),
            (409, None, (True, None, "http_409")),
            (500, None, (True, None, "http_500")),
            (503, None, (True, None, "http_503")),
            (599, None, (True, None, "http_599")),
            (400, None, (False, None, "http_400")),
            (404, None, (False, None, "http_404")),
        ],
    )
    def test_classify_by_http_status(self, status_code, retry_after_header, expected):
        """Test status-code based error classification."""
        # Arrange
        exc = Exception()
        exc.status_code = status_code
        if retry_after_header:
            exc.response = Mock(headers={"retry-after": retry_after_header})

        # Act
        result = openai_utils.classify_openai_error(exc)

        # Assert
        assert result == expected

    @pytest.mark.parametrize(
        "class_name",
        ["RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"],
    )
    def test_classify_by_known_exception_class_name(self, class_name):
        """Test classification by SDK exception class names when status code is missing."""
        # Arrange
        MockExcClass = type(class_name, (Exception,), {})
        exc = MockExcClass("SDK Error")

        # Act
        result = openai_utils.classify_openai_error(exc)

        # Assert
        assert result == (True, None, class_name)

    def test_classify_unknown_exception(self):
        """Test fallback for unhandled exceptions."""
        # Arrange
        class CustomError(Exception):
            pass

        exc = CustomError("Something broke")

        # Act
        result = openai_utils.classify_openai_error(exc)

        # Assert
        assert result == (False, None, "CustomError")


# ============================================================================
# 4. Call With Retries Tests
# ============================================================================

class TestCallWithRetries:
    """Tests function execution with backoff and retry logic."""

    def test_call_with_retries_success_first_try(self):
        """Test successful execution without retries."""
        # Arrange
        func = Mock(return_value="success_result")

        # Act
        result = openai_utils.call_with_retries(
            func,
            max_retries=3,
            retry_base_delay_sec=1.0,
            label="test_op",
        )

        # Assert
        assert result == "success_result"
        assert func.call_count == 1

    @patch("time.sleep")
    def test_call_with_retries_retryable_success(self, mock_sleep, caplog):
        """Test successful execution after encountering retryable errors."""
        # Arrange
        err_429 = Exception()
        err_429.status_code = 429
        err_429.response = Mock(headers={"retry-after": "2.5"})

        err_500 = Exception()
        err_500.status_code = 500

        func = Mock(side_effect=[err_429, err_500, "final_success"])
        logger = logging.getLogger("test_logger")

        # Act
        with caplog.at_level(logging.WARNING):
            result = openai_utils.call_with_retries(
                func,
                max_retries=3,
                retry_base_delay_sec=1.0,
                label="test_op",
                logger=logger,
            )

        # Assert
        assert result == "final_success"
        assert func.call_count == 3
        assert mock_sleep.call_args_list == [call(2.5), call(2.0)]
        assert "Retryable test_op error (http_429)" in caplog.text
        assert "Retryable test_op error (http_500)" in caplog.text

    @patch("time.sleep")
    def test_call_with_retries_max_retries_exceeded(self, mock_sleep):
        """Test raising last exception when max retries are reached."""
        # Arrange
        err_503 = Exception("Service Unavailable")
        err_503.status_code = 503
        func = Mock(side_effect=err_503)

        # Act & Assert
        with pytest.raises(Exception, match="Service Unavailable"):
            openai_utils.call_with_retries(
                func,
                max_retries=2,
                retry_base_delay_sec=0.5,
                label="test_op",
            )

        assert func.call_count == 3
        assert mock_sleep.call_count == 2

    def test_call_with_retries_non_retryable_raises_immediately(self, caplog):
        """Test non-retryable error (e.g. 400 Bad Request) fails immediately without retrying."""
        # Arrange
        err_400 = Exception("Bad Request")
        err_400.status_code = 400

        func = Mock(side_effect=err_400)
        logger = logging.getLogger("test_logger")

        # Act & Assert
        with caplog.at_level(logging.ERROR), pytest.raises(Exception, match="Bad Request"):
            openai_utils.call_with_retries(
                func,
                max_retries=3,
                retry_base_delay_sec=1.0,
                label="test_op",
                logger=logger,
            )

        assert func.call_count == 1
        assert "Non-retryable test_op error (http_400)" in caplog.text
