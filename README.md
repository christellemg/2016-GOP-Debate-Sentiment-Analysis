# Introduction et méthodologie

Dans le cadre des élections présidentielles américaines de 2016, une série de 12 débats républicains ont pris place entre les 17 candidats, entre le 6 août 2015 et le 10 mars 2016. Le premier débat, organisé par Fox News et Facebook, modéré par Bret Baier, Megyn Kelly et Chris Wallace, eut lieu à l’arène « Quicken Loans » à Cleveland, Ohio. Les dix candidats les plus populaires dans les sondages y étaient invités : Donald Trump, Jeb Bush, Scott Walker, Mike Huckabee, Ben Carson, Ted Cruz, Marco Rubio, Rand Paul, Chris Christie et John Kasich.
Très rapidement, le réseau social Twitter s’est enflammé puisque plusieurs milliers d’utilisateurs se sont exprimés sur ce débat. Quelles ont été les réactions des citoyens américains ? Est-il possible de discerner une certaine tendance d’avis compte tenu du vaste étendu d’information à disposition ?
Le but de ce projet est de faire une analyse de sentiment. En d’autres termes, il s’agit d’analyser une grande quantité de données afin de déduire les différents sentiments exprimés : positif, neutre et négatif.

Grâce à l’approche du train-test-split de la librairie Scikit-Learn, le jeu de données est divisé en ensemble d’entraînement (80%) et de test (20%). Dans un premier temps, un modèle de référence sera testé : celui du Naïve Bayes Multinomial. Il s’agit d’un modèle assez simple qui ne nécessite pas de définir de paramètres. Ensuite, deux modèles linéaires, la régression logistique et le Support Vector Classifier (SVC), ainsi que deux modèles non-linéaires, le K-Nearest Neighbor (KNN) et les forêts aléatoires (RF), sont également testés. Étant donné qu’au vu de la revue de littérature le modèle de réseaux de neurones produit des performances similaires aux autres modèles, ce modèle n’est pas considéré. Les hyperparamètres des modèles sont optimisés pour maximiser le score F1 macro. Une fois optimisés, ces modèles sont comparés au moyen du score F1 macro calculé sur l’échantillon test.

L’ensemble des performances est mesuré au moyen de l’indicateur « F1 score macro average » . Le score F1 est une mesure de performance calculée à partir des mesures de précision et de recall. L’utilisation du score F1 macro permet donc de mieux évaluer la performance du modèle, dans le cadre de classes déséquilibrées, que la mesure d’accuracy. Finalement, étant donné que le jeu de données est déséquilibré, deux méthodes de suréchantillonnage des données minoritaires, méthode « Random Oversampler » et « SMOTE » sont testées.

# Conclusion

Dans le contexte d’analyse de sentiment de tweets portant sur le premier débat de la primaire républicaine de 2016, l’analyse exploratoire a mis en évidence un déséquilibre de la variable réponse sentiment, avec 61% des tweets qui expriment un sentiment négatif, 23% un sentiment neutre et 16% un sentiment positif. L’analyse comparative de cinq modèles de classification (sans suréchantillonnage) selon le critère du score F1 macro a montré que le modèle SVC (57%) est le plus performant suivi du modèle de régression logistique (54%) et du modèle de Random Forest (54%). L’utilisation d’une méthode de suréchantillonnage a prouvé le bénéfice du rééquilibrage des classes en augmentant la performance du modèle de régression logistique de 5%, soit un score F1 macro de 59%. Finalement, la classification du sentiment neutre représente la tâche la plus difficile pour nos modèles, puisque son vocabulaire n’est pas discriminant et traduit une absence de sentiment.
Pour augmenter la performance des modèles, plusieurs améliorations peuvent être apportées :
1. Tester les autres modèles avec une méthode de suréchantillonnage
2. Utiliser un lexique prédéfini de mots associés aux sentiments dans le positif, neutre et négatif
3. Prendre en compte l’identification d’éléments informels intensifiant tels que les emojis, les majuscules et la répétition de lettres (happyyyy) etc.
4. Envisager une approche différente au problème telle que dans un premier temps, déterminer la neutralité du tweet pour ensuite déterminer la polarité du tweet le cas échéant (Négatif/Positif)
In fine, une limitation de l’utilisation de l’apprentissage automatique dans le cadre de l’analyse de sentiment réside dans la détection du sarcasme et des expressions de la langue (« he killed it ! ») qui aboutira à de mauvaises classifications. De plus, il est important de noter que le sentiment attribué aux tweets dans la base de données à l’étude a été effectué de manière subjective par un tiers parti, ce qui représente un biais intrinsèque aux données dans le cadre d’un apprentissage supervisé.
