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
Reste en P0 : 7 hémisphères dont la piale ne démarre pas, 2 sous le seuil de distance. Cause des 7 (octobre
2026, reproduction sur sub-re8lj) : dès la blanche initiale, des sommets « plissés » sans cône de décalage
(aucune direction à angle aigu avec toutes les faces incidentes ; ex. valence 12, somme des angles 582°, faces
incidentes opposées, triangles corrects) ; tout décalage, si petit soit-il, croise leurs propres faces, et la
réduction du décalage ne peut qu'échouer ; les lames vues ensuite sont fabriquées par les tentatives de
sauvetage. Ni le déplacement du sommet vers le barycentre, ni une détente laplacienne sur 4 anneaux (même sans
garde-fou) ne rendent de cône : il faudrait un remaillage local du pli (ou une correction dans le masque) ; quelques piales héritent d'une blanche déjà serrée (dégagement < 0,1 mm) ; côté gyroxide,
masque médial lu depuis `_cortex.shape.gii`, sortie GIfTI en mm et pas de temps à dériver de la qualité
du maillage. Retour du collègue (DPF*) : améliorer l'extraction de la blanche (ondulations millimétriques
avant 28 SA) ; α_ref à recalibrer pour les fœtus. Essai non retenu (octobre 2026, branche locale
`lissage-contraint`) : lissage laplacien contraint à une bande de ±0,8 voxel autour de la frontière de l'objet
blanc ; il divise par deux les taches de la DPF* à 21 SA (HD95 0,17 → 0,30 mm) mais amincit les crêtes
étroites en lames que le remaillage replie ou que la piale ne peut quitter (validation 8 : 379/386 et
374/384 ; après remaillage, nouveaux échecs dès 5 pas). Un flou plus fort de la distance signée ferme les
sillons (HD95 0,8 mm à 36 SA) ; plus de pas de Taubin ne lisse presque rien. Pistes restantes : laplacien
épargnant les sommets de forte courbure, ou fonction implicite plus lisse (probabilités de la segmentation).

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
(repères tracés sur des surfaces dHCP release 1, disponibles au labo mais pas encore sur Garage, et KKI) ; avant 28 SA, α_ref = 500 donne un noyau
de ~1,2 mm et la DPF* capte les ondulations millimétriques du contour de la segmentation (présentes sur la
blanche à toutes les échelles, pas un défaut du maillage).

Baseline de référence pour P0 : fetpype `surface_processing` (github.com/fetpype/surface_processing).

## Conventions (alignées sur medoxide)

- Workspace Cargo ; dépendances partagées dans `[workspace.dependencies]`, `Cargo.lock` versionné.
- Burn 0.22 (`default-features = false`) ; même version que medoxide pour un futur binaire commun.
- CPU par défaut avec le backend `flex` (déterministe, utilisé par les tests) ; GPU par les features
  `cuda` (PC Linux RTX 2080 Ti) ou `wgpu` (Mac).
- `surfoxide-core` reste sans Burn ni format de fichier ; un crate par module ; `surfx` est la CLI.
- Pas de dépendance sans code qui l'utilise.
- Commentaires, documentation, messages d'erreur et noms de tests en français.
- Avant chaque commit : `cargo fmt`, `cargo clippy --all-targets`, `cargo test --release`.

## Machines

- Développement (CPU et GPU) : PC Linux, Ubuntu x86_64, RTX 2080 Ti (11 Go, Turing sm_75 : f32 et f16,
  pas de bf16), 12 cœurs, 64 Go. Installation : `scripts/setup-linux.sh` (CUDA Toolkit seul, version
  plafonnée par le pilote).
- Machine WSL (RTX 3080, alias ssh `sshwsl`) : secondaire, pas toujours allumée ; `scripts/setup-wsl.sh`.
- La compilation `--features cuda` télécharge un paquet LLVM (~45 Mo) dans `~/.cache/tracel/`.
- Données réelles (hors Git) : bucket `garage:fetal-crl2025`, dérivées dans
  `derivatives/fetal-folding-geometry/<sub>/<ses>/anat/` (BOUNTI-19, multi-BOUNTI-43, T2w rééchantillonnée
  à 0,5 mm). `bash scripts/fetch-data.sh` les rapatrie dans `$SURFOXIDE_DATA` (défaut
  `~/.cache/surfoxide/data`) pour les tests ignorés, la validation et les figures. Âges gestationnels :
  `~/Exp/fetal-folding-geometry/data/subjects.csv`. Atlas fœtal dHCP (GIfTI) : `~/Exp/gyroxide/data/atlas_dhcp/`.
