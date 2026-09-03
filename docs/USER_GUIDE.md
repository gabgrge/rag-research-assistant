# Guide utilisateur

## Démarrer l'application

- Ouvrir le menu Démarrer.
- Cliquer sur "Assistant documentaire".

Si le raccourci n'existe pas, demander à l'installateur de le recréer.

## Écran principal

L'écran principal comporte :

- Une zone de conversation.
- Une barre latérale avec les paramètres.

## Mode de réponse

Deux modes sont disponibles :

- Recherche: réponse factuelle avec citations.
- Résumé: réponse synthétique et structurée.

Choisir le mode selon l'objectif de la question.

## Passages par document

Ce réglage évite qu'un seul document domine la réponse. Plus la valeur est haute, plus un même document peut contribuer à la réponse.

## Filtres

Les filtres permettent de limiter la recherche :

- Nature (dossier ou catégorie)
- Type de fichier
- Nom du fichier

Utiliser "Réinitialiser les filtres" pour revenir à une recherche globale.

## Poser une question

1. Saisir la question dans la zone de chat.
2. Valider pour lancer la recherche.
3. Lire la réponse et consulter les sources si besoin.

## Voir les sources

Sous chaque réponse, un panneau "Sources" affiche :

- Le document d'origine.
- La page ou le paragraphe concerné.

## Mise à jour de la base documentaire

1. Ouvrir l'onglet "Mise à jour".
2. Cliquer sur "Mettre à jour".
3. Confirmer.

La mise à jour peut prendre plusieurs minutes et utiliser des crédits API. Ne pas fermer l'application pendant l'opération.

## Quitter l'application

Utiliser le bouton "Quitter l'application" dans la barre latérale.
Fermer le navigateur ne stoppe pas l'application.

## Bonnes pratiques

- Poser des questions précises.
- Utiliser les filtres pour réduire le bruit.
- Mettre à jour la base après ajout de documents.

## Dépannage rapide

- Message "Je ne trouve pas de passage pertinent" : reformuler la question ou utiliser des filtres.
- Erreur RAG : vérifier la connexion Internet et la clé API dans `.env`.
- Échec de mise à jour : vérifier la synchronisation Google Drive et l'installation de LibreOffice.
