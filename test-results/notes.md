## Principes

<p>J'ai effectué plusieurs tests sur LLaVA Next, en comparant les deux modèles les plus performants : Vicuna 13b et Hermes-Yi 34b. Pour l'instant, 3 tests ont été effectués sur un tableur exemplaire :</p>

- Résumé du tableur
- Interprétation du tableur
- Conversion en csv, en vue de faire du traitement de données

<p>Cette image contient plusieurs "pièges" :</p>

- Toutes les valeurs de "In" sont à 5, sauf pour juin, valant 6 : ceci a pour but de *casser le motif que le modèle pourrait supposer, et détecter s'il le repère*.
- Toutes les valeurs de "Out" sont à 1, sauf pour mars et août, valant 2, *dans le même but que précédemment*.
- Toutes les valeurs de "Savings" sont calculées à partir de "In" - "Out", sauf en janvier, valant 1 : ceci a pour but de *sonder les capacités de raisonnement mathématique du modèle, et a capacité à repérer des incohérences*.

<p>J'ai fait varier la résolution des images pour sonder son impact sur la qualité des réponses :</p>

| **Base**  |**Med-res**| **High-res**| **Adapté**  |
|-----------|-----------|-------------|-------------|
| 362 x 248 | 805 x 544 | 1610 x 1088 | 1600 x 1088 |

<p>La motivation derrière la résolution adaptée est que le modèle ViT sur lequel se base CLIP pour encoder les images, les divise en patch de 16 x 16 dans le cas de LLaVA. La résolution adaptée pourrait potentiellement permettre un meilleur traitement des données, sans fine-tuning.</p>

## Résumer

<p>Ce test a pour but de voir si le modèle peut "at face value" comprendre ce qu'il voit dans les grandes lignes, à travers le prompt : "Describe each row and column of this spreadsheet."</p>
- Vicuna résume de manière adéquate le document. Il rend :
  - Le nombre de colonnes du document
  - Le nombre de lignes du document
  - Une explication détaillée des colonnes et des lignes, avec quelques exemples de leur contenu
- Hermes produit un rendu semblable à celui de Vicuna. Il est plus précis et plus concis que Vicuna.

<p>A noter que le modèle Vicuna a essayé de donner une liste des valeurs trouvées, rendant un équivalent de 14 mois, tout en n'ayant aucunement repéré les valeurs pièges. De plus, pour l'image de base, le modèle a affirmé avoir vu 6 colonnes.  

Le modèle Hermes affirme voir des données de différentes couleurs pour la résolution de base. Avec la résolution moyenne, le modèle affirme voir des variations des valeurs des colonnes, mais donne des valeurs aberrantes et incohérentes : des épargnes de 6, des entrées de 1, ...

Remarquablement, pour les images HD et adaptée, Hermes restitue toutes les données du tableur, incluant les valeurs "pièges", et corrige l'erreur de calcul pour la valeur de "Savings" en janvier.</p>

<p>**Conclusion** : la performance semble correcte, *en moyenne*. La compréhension semble superficielle, à part pour Hermes dans les hautes résolutions.</p>


## Interprétation

<p>Pour qualifier plus rigoureusement la compréhension de l'image par le modèle, j'estime qu'il faut le sonder de manière plus détaillée. La capacité du modèle à interpréter les données est évaluée à travers deux prompts principaux :</p>
- "What does the [nth] column correspond to?"
- "What does the [nth] row correspond to?"

<p>Je juge ensuite manuellement les réponses des modèles.</p>

|        | Précision colonnes | Précision lignes | Hallucination | Résolution          |
|--------|--------------------|------------------|---------------|---------------------|
| Vicuna | 45%                | 35%              | 100%          | Aucun effet notable |
| Hermes | 45%                | 90%              | 100%          | Aucun effet notable |

<p>Les tendances remarquables :</p>
- Lorsqu'une information sur une ligne/colonne inexistante est demandée, les deux modèles vont halluciner lesdites lignes/colonnes à coup sûr.
- Il y a une confusion quasi-constante (87% du temps) sur les modèles, confondant les 2èmes et 3èmes colonnes pour la 4ème.

<p>**Conclusion** : quand demandé, les deux modèles restituent une compréhension très moyenne de la structure des données présentées, même pour un cas simple tel que cette image. <span style="color: blue;">*[Une explication plausible du problème ? (lien)](https://arxiv.org/abs/2211.11153)*</span></p>


## Conversion en csv

<p>Pour qualifier la capacité du modèle à restituer les informations qu'il trouve, je lui demande de transformer l'image en fichier csv, avec le prompt : "Convert this image into a .csv file:"</p>

|                 | **Différence colonnes** | **Différence lignes** | **Précision** |
|----------------:|--------------------------|------------------------|---------------|
| **Vicuna** base | 0                        | +62                    | 75%           |
|             med | 0                        | +62                    | 56%           |
|            high | 0                        | +62                    | 56%           |
|         adapted | 0                        | +62                    | 56%           |
| **Hermes** base | 0                        | 0                      | 75%           |
|             med | 0                        | 0                      | 78%           |
|            high | 0                        | +2                     | 100%          |
|         adapted | 0                        | 0                      | 16%           |

<p>Les 12 premières lignes et 3 premières colonnes ont été sélectionnées pour réaliser l'évaluation de la précision, en considérant que les colonnes/lignes supplémentaires sont hallucinées.</p>

- **Vicuna**: Incapable de marquer la fin du fichier, le modèle ne s'arrête de générer que quand il atteint le nombre maximal de tokens en sortie (explicant les 62 lignes en trop). En moyenne, 1 erreur sur 2 est commise dans la catégorie "Savings", le reste étant équiréparti entre "In" et "Out".
- **Hermes**: Modèle plus hétérogène selon la résolution de l'image d'entrée, avec d'excellentes capacités de restitution, notamment pour le fichier HD, hallucinant cependant 2 lignes. La répartition des erreurs est la même que pour Vicuna. Ceci pourrait pointer vers un défaut de la projection CLIP -> LLM plutôt que le LLM en tant que tel.

<p>**Conclusion** : En partant du principe que le tableur donné produit des résultats représentatifs pour un tableur simple quelconque :</p>
<p>Vicuna, modèle capable de tourner sur un GPU standard 1080 Ti, est plus adapté pour le résumé de documents élémentaires, que leur compréhension précise (tâche pour laquelle il semble mal adapté).
Ses performances pour résumer un tableur sont sensiblement comparables à celles de Hermes.</p>

</p>Hermes, plus lourd et plus complexe, semble être bien plus adapté pour la compréhension/interprétaiton de tableurs, étant plus précis dans les informations qu'il restitue. Ce modèle semble être capable d'halluciner sur des tâches basiques, et est plus sensible à la résolution de de l'image, performant très bien à haute résolution.</p>