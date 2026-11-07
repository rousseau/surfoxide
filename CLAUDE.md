# surfoxide — instructions pour Claude Code

## Le projet

Boîte à outils Rust/Burn pour les surfaces corticales (IRM fœtale et néonatale). Trois dépôts liés
par les données, pas par le code :

- **medoxide** (github.com/rousseau/medoxide) : reconstruction SVR, segmentation (multi-)BOUNTI → NIfTI.
- **surfoxide** (ce dépôt) : surfaces WM/GM et GM/CSF, contrat de validité, mesures corticales → GIfTI.
- **gyroxide** : tétraédrisation, champs issus des images, simulation du plissement ; renvoie sa
  surface simulée à surfoxide pour qu'elle soit mesurée par le même code que les surfaces réelles.

Périmètre : **surfoxide ne traite que des surfaces** (triangulaires, et champs par sommet). Aucun
maillage volumique ici : la génération des maillages tétraédriques appartient à gyroxide, à partir
des surfaces livrées par surfoxide. Ne pas ajouter de tétraédrisation ni de lecture `.vtu` dans ce dépôt.

Document de référence (état de l'art, positionnement, priorités) :
https://claude.ai/code/artifact/492fa76d-b8bf-4ad7-bf59-796739415874

## Contrat de données

- Coordonnées monde RAS en mm ; affine de l'image source conservée dans les métadonnées.
- Faces orientées, normales sortantes (`TriMesh::signed_volume() > 0` pour une surface fermée).
- Formats : NIfTI en entrée, GIfTI pour surfaces et champs par sommet. Le `.vtu` (volumique) reste
  côté gyroxide ; ses surfaces simulées reviennent sous forme de surfaces (GIfTI ou VTK PolyData).
- Une surface livrée à gyroxide est : fermée, 2-variété, orientée de façon cohérente, d'un seul
  tenant, de genre 0, sans auto-intersection ; la WM est strictement dans la piale avec une
  épaisseur minimale.

## Priorités (voir le document de référence)

P0, des surfaces utilisables par gyroxide :
1. Types de base (`TriMesh`, `Volume`, champs par sommet, conversion vers tenseurs Burn) — fait.
2. Entrées-sorties : NIfTI (segmentations), GIfTI, VTK `.vtp`, PLY/STL, surfaces FreeSurfer — fait.
3. Masque d'hémisphère fermé à partir des labels BOUNTI (19 régions) et multi-BOUNTI — fait.
4. Correction topologique voxelique (Bazin, fast marching) — fait.
5. Marching cubes à topologie cohérente, lissage de Taubin — fait.
6. Remaillage isotrope — fait.
7. Contrôles géométriques : topologie, auto-intersections (AABB + tests triangle–triangle exacts),
   distance minimale WM–piale — fait (`surfoxide-check`).
8. Réparation locale des faces fautives — fait (déplacement de sommets uniquement).
9. Piale sans apprentissage : gonflement de la WM vers la frontière cortex/LCR (flot déterministe) — fait :
   lignes de courant du champ de Laplace, sillons enfouis ouverts par leur nappe médiane.
Validation sur toute la cohorte (octobre 2026, `scripts/validate.py --all`, 193 sessions de 20 à 38 SA) :
contrat complet (blanche + piale + distance minimale) 383/386 hémisphères BOUNTI-19 (99,2 %) et 378/384
multi-BOUNTI-43 (98,4 %) ; HD95 médian 0,17 et 0,19 mm ; épaisseur corticale médiane (longueur des lignes
de courant de Laplace, face médiale exclue) 1,70 et 1,49 mm.
Comparaison à fetpype (19 sessions) : blanche conforme 46/46 contre 24/38, HD95 0,18 contre 0,40 mm.
Présentation : https://claude.ai/artifact/DVcshJu5TQaxDFMLKSLWdZ
Piale utilisable par gyroxide (octobre 2026) : dégagement imposé entre berges des sillons (`min_clearance`
= 0,25 × arête moyenne de la blanche ; sans lui, l'auto-contact de gyroxide agit dès l'état initial et la
simulation diverge) et régularisation tangentielle des triangles (`surfoxide_pial::regularize`, même
triangulation que la blanche). `surfx check` rapporte `min_clearance`. Validation 7 : contrat inchangé
(383/386, 378/384) ; ~18 % des sommets de piale retenus (manque moyen 0,45 mm, sans effet sur l'épaisseur).
`safe_update_with` ne recontrôle, après le premier passage, que les faces des sommets ramenés (résultat
identique au bit près, vérifié sur 21 hémisphères) : piale à 36 SA 163 → 90 s. Boucle de bout en bout faite sur une session de 21 SA : `gyroxide/tools/atlas_to_mesh.py`
(TetGen) sur notre piale + blanche, simulation complète (il faut `--mesh-spacing 0.0136`, le `a` de l'atlas :
avec l'arête moyenne, divergence vers t ≈ 0,1), relecture et DPF* ici ; la surface simulée a des
auto-intersections (contact par pénalité), que `surfx check` signale.
Piales qui ne démarraient pas (7 hémisphères de la validation 7) : dès la blanche initiale, des sommets
« plissés » sans cône de décalage (aucune direction à angle aigu avec toutes les faces incidentes ; ex. valence
12, somme des angles 582°, faces incidentes opposées, triangles corrects) ; tout décalage croise leurs propres
faces. Remède (novembre 2026) : seconde passe de la piale dans `surfx run`, seulement si la première (la
séquence de reprises de la validation 7, inchangée) échoue ou finit sous l'épaisseur minimale : blanche
initiale reprise par `surfoxide_remesh::open_cones` (bascules d'arêtes puis fusion d'arête sous link
condition, retenues si elles relèvent le pire cône du voisinage sans y ajouter d'auto-intersection), puis
`PialParams::cone_axis` (axe `vec3::cone_axis`, max-min des cosinus, quand la recherche depuis la normale
échoue) ; on garde la meilleure des deux passes. Mettre `open_cones` en premier dégradait 11 hémisphères
(décalages quasi tangents, distance minimale ~0). Validation 10 : 386/386 BOUNTI-19 et 382/384 multi-BOUNTI,
aucun hémisphère perdu, les 761 déjà conformes identiques au bit près ; HD95 médian inchangé (0,17 et 0,19 mm).
Échecs connus (acceptés) : sub-ru8l4 ses-03 L, segmentation signalée par `segqc` (304 mm² de substance blanche
au contact du fond), triangles quasi dégénérés autour du trou ; sub-wediw ses-01 R, distance minimale
0,009 mm (la seconde passe ne fait pas mieux). Reste en P0 : quelques piales héritent d'une blanche déjà
serrée (dégagement < 0,1 mm). Micro-arêtes (novembre 2026, benchmark de la suite sur trois machines) : la détente des sommets aigus de
la blanche (`relax_sharp_vertices`, laplacien local répété 20 fois) contractait parfois un voisinage en amas
d'arêtes de 0,02 mm, que la piale héritait (arête min 0,011 mm, dégagement 0,003 mm) ; la plus courte arête de
la piale fixe le pas de temps de gyroxide (12 fois plus de pas). Remède : fusion des arêtes de moins d'un quart
de l'arête visée (`clean_short_edges`, seuil de la régularisation de la piale) après la détente ; sur ce cas,
arête min de la piale 0,133 mm et dégagement 0,129 mm, l'autre hémisphère identique au bit près. Validation 12
(PGX, contre la 10) : 386/386 et 383/384 (sub-wediw ses-01 R gagné, aucun perdu), 369 et 230 rapports identiques,
HD95 et épaisseur médians inchangés ; piales dont l'arête min est < 0,05 mm : 1 → 1 (BOUNTI-19) et 69 → 18
(multi-BOUNTI) ; 6 hémisphères multi-BOUNTI dégradés (arête min jusqu'à 0,003 mm, contrat respecté). Diagnostic
(SURFOXIDE_DEBUG, 19 cas) : 17 arêtes courtes venaient d'amas de micro-arêtes de la blanche que la link condition
empêche de fusionner, les autres du décalage initial dans des concavités serrées (normales convergentes) ; la
progression ne raccourcit rien (dégagement minimal = seuil d'arête courte) et la régularisation ne bouge pas les
sommets figés sous la distance minimale. Remède (novembre 2026) : `surfoxide_remesh::untangle_short_edges` sur la
blanche après la détente et après chaque reprise (bascule ou fusion voisine puis fusion de l'arête courte, retenue
si elle relève l'arête min locale sans abîmer cône ni intersections) ; décalage initial réduit aux extrémités des
arêtes qu'il raccourcit sous le seuil (plancher 1,5 × distance minimale) ; repli sur la blanche d'avant le
démêlage si la piale finit sous l'épaisseur minimale (sub-rag2a ses-01 R). Validation 13 (PGX, contre la 12,
`scripts/compare_validations.py`) : 386/386 et 383/384, aucun perdu, 370 et 266 rapports identiques, HD95 et
épaisseur médians inchangés ; piales d'arête min < 0,05 mm : 1 → 0 et 18 → 1 (sub-ru8l4 ses-03 L, hors contrat),
< 0,1 mm : 3 → 1 et 36 → 14 (q1 multi-BOUNTI 0,013 → 0,076 mm) ; dégagement en baisse sur 2 hémisphères, contrat
respecté (sub-5ksd1 ses-01 L BOUNTI-19 0,014 → 0,0003 mm, sub-jkq1g ses-02 R 0,129 → 0,044 mm). Restent entre
0,05 et 0,1 mm des arêtes de la blanche que le démêlage ne retire pas (kvbr9, z067x) et des plis si aigus que
l'arête diminue de moitié même au plancher du décalage (yp2at, srxv8) ; piste : fusion d'arête simultanée sur la
blanche et la piale. Essai dans gyroxide (novembre 2026, `scripts/gyroxide_check.py`, PGX, gyroxide
`main` fusionné avec la PR #9, 13 hémisphères de la validation 13, 20 000 pas en croissance uniforme et suivant
`cortex`) : le nombre de pas jusqu'à t = 1 suit l'inverse de l'arête min de la piale (37 SA multi-BOUNTI : 0,79 M
pas pour le témoin à 0,13 mm, ~25 h ; 1,31 M pour yp2at à 0,070 mm ; 1,87 M pour kvbr9 à 0,050 mm). Le faible
dégagement (5ksd1, b7x2q : 0,0003 mm, 1 contact au départ) ne gêne pas sur ces 20 000 pas. Les deux mêmes
modes de croissance se comportent pareil. Trois runs divergent (NaN : s8nzq dans les deux modes, yp2at en
croissance uniforme) : le NaN naît chaque fois dans le pire tétraèdre de TetGen, plat (qualité < 1e-4) et à
quatre sommets sur la piale, que l'option `Y` (triangles de la piale conservés) empêche de retirer ; la piale n'y est
pas serrée. Fusion couplée des arêtes courtes (`surfoxide-remesh::paired`, appelée dans `finish_pial` sous
0,2 × l'arête moyenne) : même opération (fusion, bascule ou fusion voisine préalable, deux opérations si la link
condition l'exige) sur la piale et la blanche, sommet fusionné de la piale éventuellement replacé sur la normale de
la blanche ; retenue si elle allonge l'arête locale sans dégrader blanche, cônes, intersections, distance ni
dégagement. Validation 15 (contre la 13) : contrat inchangé, aucun hémisphère dégradé, HD95 et épaisseur
médians inchangés ; piales < 0,1 mm : 1 → 0 et 14 → 5 (q0 multi-BOUNTI 0,013 → 0,050, q1 0,076 → 0,090 mm),
temps total inchangé (la validation 14, critères coûteux évalués à chaque essai, multipliait jusqu'à ×9 le
temps des hémisphères à arêtes courtes). Restent kvbr9 (0,050), z067x (0,076), ru8l4 ses-03 (0,078, hors
contrat), ru8l4 ses-02 (0,089), yp2at (0,090) : plis locaux où la piale touche presque la blanche (0,0002 à
0,003 mm) ; toute fusion y croise les deux surfaces (diagnostic sous `SURFOXIDE_DEBUG`). Piste : déplisser la
blanche en amont. Côté gyroxide (`origin/main`), sortie GIfTI en mm et pas de temps tiré de la
plus courte arête faits ; croissance suivant le champ `cortex` du `.vtu` (`--growth-field cortex`, 0 sur la
face médiale) : rousseau/gyroxide#9, non essayée sur une session réelle. Retour du collègue (DPF*) : améliorer l'extraction de la blanche (ondulations millimétriques
avant 28 SA) ; α_ref à recalibrer pour les fœtus. Essai non retenu (octobre 2026, branche locale
`lissage-contraint`) : lissage laplacien contraint à une bande de ±0,8 voxel autour de la frontière de l'objet
blanc ; il divise par deux les taches de la DPF* à 21 SA (HD95 0,17 → 0,30 mm) mais amincit les crêtes
étroites en lames que le remaillage replie ou que la piale ne peut quitter (validation 8 : 379/386 et
374/384 ; après remaillage, nouveaux échecs dès 5 pas). Un flou plus fort de la distance signée ferme les
sillons (HD95 0,8 mm à 36 SA) ; plus de pas de Taubin ne lisse presque rien. Retenu en option (novembre
2026, `surfx run --band-in 0.25`, désactivée par défaut) : bande asymétrique, +0,8 voxel à l'extérieur et
0,25 à l'intérieur (les crêtes ne sont plus amincies), 50 pas laplaciens. Validation 11 contre 10 : avant
28 SA, contrat inchangé (118/118, 116/116), petites taches de DPF* 37 → 14 et bassins 16 → 11 (médianes) ;
sur toute la cohorte 385/386 et 380/384 (4 perdus, tous ≥ 28 SA ; wediw R gagné), HD95 0,17 → 0,28 mm,
épaisseur −3 %. À réserver à la DPF* des jeunes fœtus ; `surfx run` ne connaît pas l'âge. Pistes restantes :
laplacien épargnant les sommets de forte courbure, ou fonction implicite plus lisse (probabilités de la
segmentation).

P1 : intégrateur ODE dans Burn, réseaux de déformation à la CoSeg (faiblement supervisés par les
segmentations de medoxide), opérateurs différentiels, métriques (épaisseur, profondeur sulcale,
gyrification, métrique spectrale), projection des lobes multi-BOUNTI, lecture des surfaces simulées
renvoyées par gyroxide.
Plan P1 validé (octobre 2026) : piale apprise à partir de notre blanche (déformation par ODE à la CortexODE,
supervision sommet à sommet par la piale du flot + perte faible de distance à la frontière à sillons ouverts),
entrée = segmentation seule, cohorte CRL2025 découpée par sujet (`subjects.csv`, colonne `split`) ; la sortie
repasse par nos garde-fous (contrat). Jalons : 1. opérateurs Burn ; 2. intégrateur ODE ; 3. précalcul des
données ; 4. entraînement et évaluation ; 5. intégration dans `surfx run`.
Jalon 1 — fait (`surfoxide-burn::geometry`) : normales, aires, raideur cotangente et masse appliquées à un champ,
échantillonnage trilinéaire différentiable, DPF par gradient conjugué ; égaux aux opérateurs CPU (tests `flex`),
gradients vérifiés par différences finies. Sur la 2080 Ti (blanche de 121 000 sommets) : raideur·u 2,8 ms, DPF
0,33 s. Les sommes par sommet passent par une lecture indexée au format ELL : `select_assign` avec addition est
sérialisé sur CUDA (~100 ms par appel). Temps GPU : `cargo test --release -p surfoxide-burn --features cuda --
--ignored` avec `SURFOXIDE_TEST_SURFACE=<blanche.surf.gii>`.
Jalon 2 — fait (`surfoxide-ode`) : Euler et RK4 à pas fixe sur `[N, 3]`, rétropropagation à travers les pas ;
ordres de convergence vérifiés (1 et 4), gradient exact sur une dilatation, icosphère déformée restée valide.
Sur la 2080 Ti : 121 000 sommets, 10 pas RK4 dans un champ échantillonné, passe avant + arrière 2,25 s, 2,6 Go.
Jalon 3 — fait (`surfx dataset`, `scripts/dataset.py`) : 761 hémisphères conformes de la validation 7 (deux tables
de labels) dans `~/scratch/surfoxide-dataset` (5,2 Go, hors Git ; `index.tsv` : 455 entraînement, 143
validation, 163 test, folds par sujet). Par exemple : classes `u8` sur le cadrage de l'hémisphère à 0,5 mm
(0 fond, 1 cortex, 2 substance blanche fermée, 3 nappe médiane des sillons enfouis), distance signée `f32` aux
faces des voxels de la frontière à sillons ouverts (corrigée d'un demi-voxel), blanche et piale de même
triangulation. Contrôle : blanche à −1,0 à −1,6 mm, piale à +0,14 à +0,20 mm de cette frontière (médianes).
Jalon 4 — en cours (`surfoxide-learn`, `surfx train`) : encodeur CNN 3D à trois échelles (0,5/1/2 mm) sur les
classes, MLP de vitesse sur les caractéristiques échantillonnées et le temps, 10 pas RK4 depuis la blanche ;
pertes L2 aux sommets de la piale + distance signée au carré ; Adam ; entraînement sur un tirage de 16 384
sommets par pas (chaque sommet suit sa propre trajectoire). Surapprentissage contrôlé (2 hémisphères, 150
époques, 2080 Ti) : erreur aux sommets 1,38 → 0,16 mm, distance à la frontière 1,07 → 0,12 mm.
**P1 en attente (24 octobre 2026, décision de l'utilisateur : pas prioritaire).** Entraînement complet non mené :
première version hors mémoire sur les grands hémisphères (convolutions à 0,5 mm) ; version corrigée (classes
lues telles quelles à 0,5 mm, convolutions à 1/2/4 mm) : 10,2 Go et GPU à 98 %, plus d'une heure par époque de
455 exemples (16 384 sommets, 10 pas RK4), soit plus de 20 h pour 20 époques. Pour reprendre : de préférence sur
la station PGX (128 Go de mémoire unifiée), sinon sur la 2080 Ti avec `--vertices 8192 --ode-steps 6` ; commande
`surfx train --dataset ~/scratch/surfoxide-dataset --out <dir> --backend cuda` (jeu à régénérer par
`scripts/dataset.py` si `~/scratch` a été vidé). Puis évaluation sur le fold de test contre le flot actuel
(contrat après garde-fous, HD95, épaisseur, retrait au fond des sillons, temps).
Profondeur sulcale — fait (`surfoxide-metrics`, `surfx depth`, écrite par `surfx run` sur la blanche) : DPF*
(Dieudonné, Auzias, Lefèvre, Comput Biol Med 2026), pas le `sulc` de FreeSurfer (dépendant de la taille).
Choix confirmés par les auteurs (code : github.com/maximedieudonne/DPF-star) : système (S/2 + αB) D = 2B(K − K̄)
(choix de formalisation), α_ref = 500, L = V^(1/3) avec V au choix volume de l'enveloppe convexe (défaut, comme
leur code) ou volume enfermé (`--length volume`, comme le texte de l'article) ; D* sans dimension (le ×100 de
leur code n'est que d'affichage, non repris) ; signe conventionnel, crêtes positives. Vérifiée contre ce code
sur l'atlas dHCP 21/28/36 SA (courbure à 6e-8, D* à 6e-6 près pour un IPR de 0,03 à 0,07) ; invariance par
homothétie et robustesse au remaillage (r ≥ 0,987 de 0,35 à 1 mm) testées. Points ouverts : benchmark Zenodo
(repères tracés sur des surfaces dHCP release 1, disponibles au labo mais pas encore sur Garage, et KKI ;
`scripts/dpf_benchmark.py` calcule StdCrest, Sep et Dev à partir des repères, autotest sur une sphère ondulée) ; avant 28 SA, α_ref = 500 donne un noyau
de ~1,2 mm et la DPF* capte les ondulations millimétriques du contour de la segmentation (présentes sur la
blanche à toutes les échelles, pas un défaut du maillage). Comptés en bassins (D* < 0) d'au moins 5 mm², et non en
sommets, nos blanches et l'atlas sont comparables avant 25 SA à α_ref = 500 (10 contre 8 à 13) ; nous en comptons
~1,5× plus après 30 SA. Note de discussion pour les auteurs (options α_ref = 500, α_ref fœtal, noyau plancher
L/√α ≥ 1,65 mm) : https://claude.ai/artifact/Li49SnNzY9pPMVVZUvEuB2

Cohérence BOUNTI-19 / multi-BOUNTI-43 (octobre 2026, validation 7, 384 paires d'hémisphères) : volume de la
substance blanche robuste (CCC 0,999), mais multi-BOUNTI donne une blanche plus plissée (aire +12 %, indice de
gyrification +10 % après 35 SA), un cortex plus mince (−11,5 %, CCC 0,19) et plus de bassins de DPF* (+28 %
après 35 SA) : ne jamais mélanger les deux tables dans une analyse. Des sessions multi-BOUNTI ont du tissu
étiqueté fond (sub-vm989, trois sessions, jusqu'à 42 mL ; sub-14k3w, sub-hkay8 ; 18 sessions > 3 mL), vérifié
sur la T2w ; signalé à medoxide (rousseau/medoxide#10). Contrôle intégré : `surfoxide_mask::segmentation_qc`,
`surfx segqc`, et `segmentation` dans le rapport de `surfx run` ; suspect si la substance blanche touche le fond
sur plus de 100 mm² par hémisphère, ou si plus de 15 % du bord externe du cortex touche le fond au lieu du LCR
(cortex perdu). Sur la cohorte : 18/18 sessions > 3 mL signalées, aucun hémisphère BOUNTI-19.

Baseline de référence pour P0 : fetpype `surface_processing` (github.com/fetpype/surface_processing).

## Conventions (alignées sur medoxide)

- Workspace Cargo ; dépendances partagées dans `[workspace.dependencies]`, `Cargo.lock` versionné.
- Burn 0.22 (`default-features = false`) ; même version que medoxide pour un futur binaire commun.
- CPU par défaut avec le backend `flex` (déterministe, utilisé par les tests) ; GPU par les features
  `cuda` (NVIDIA) ou `wgpu` (Metal sur Mac, Vulkan sous Linux).
- `surfoxide-core` reste sans Burn ni format de fichier ; un crate par module ; `surfx` est la CLI.
- Pas de dépendance sans code qui l'utilise.
- Commentaires, documentation, messages d'erreur et noms de tests en français.
- Avant chaque commit : `cargo fmt`, `cargo clippy --all-targets`, `cargo test --release`.

## Environnement

- Le code doit compiler et passer ses tests sur macOS arm64, Linux x86_64 et Linux aarch64. Les machines,
  leurs accès et la règle de passage de l'une à l'autre sont décrits hors de ce dépôt.
- `--features cuda` télécharge au premier build un paquet LLVM de Tracel (~45 Mo). Sous Linux aarch64, lier
  avec `lld` (`-C link-arg=-fuse-ld=lld`) : le lieur GNU ne résout pas les symboles de ce LLVM statique.
- Installation d'un PC Linux x86_64 avec GPU NVIDIA : `scripts/setup-linux.sh` (CUDA Toolkit seul, version
  plafonnée par le pilote) ; sous WSL : `scripts/setup-wsl.sh`.
- Données réelles (hors Git), sur Garage sous `$OXIDE_REMOTE` (remote rclone et bucket, définis hors du dépôt,
  en général dans `~/.config/garage/env`) : `derivatives/fetal-folding-geometry/<sub>/<ses>/anat/`
  (BOUNTI-19, multi-BOUNTI-43, T2w rééchantillonnée à 0,5 mm). `bash scripts/fetch-data.sh` les rapatrie dans
  `$SURFOXIDE_DATA` (défaut `~/.cache/surfoxide/data`) pour les tests ignorés, la validation et les figures.
- Résultats de travail partagés entre machines (validations, jeu d'entraînement de P1, essais) :
  `$OXIDE_REMOTE/derivatives/surfoxide/<dossier>`, miroir de `~/scratch/<dossier>` (ou `$SURFOXIDE_SCRATCH`) ;
  `bash scripts/sync-scratch.sh list | pull <dossier>… | push <dossier>…` (copie sans suppression, jamais les
  `target-*`). Toute nouvelle validation ou tout nouveau jeu se pousse une fois terminé.
- Âges gestationnels et folds : `subjects.csv`, rangé sur Garage avec les segmentations et rapatrié avec elles
  dans `$SURFOXIDE_DATA` (`$OXIDE_SUBJECTS` pour un autre fichier).
  Atlas fœtal dHCP (GIfTI) : `$OXIDE_DHCP_ATLAS` (défaut `data/atlas_dhcp/` du clone de gyroxide voisin).
