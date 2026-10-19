# surfoxide — instructions pour Claude Code

## Le projet

Boîte à outils Rust/Burn pour les surfaces corticales (IRM fœtale et néonatale). Trois dépôts liés
par les données, pas par le code :

- **medoxide** (github.com/rousseau/medoxide) : reconstruction SVR, segmentation (multi-)BOUNTI → NIfTI.
- **surfoxide** (ce dépôt) : surfaces WM/GM et GM/CSF, contrat de validité, mesures corticales → GIfTI.
- **gyroxide** : tétraédrisation, champs issus des images, simulation du plissement ; renvoie sa
  surface simulée à surfoxide pour qu'elle soit mesurée par le même code que les surfaces réelles.

Document de référence (état de l'art, positionnement, priorités) :
https://claude.ai/code/artifact/492fa76d-b8bf-4ad7-bf59-796739415874

## Contrat de données

- Coordonnées monde RAS en mm ; affine de l'image source conservée dans les métadonnées.
- Faces orientées, normales sortantes (`TriMesh::signed_volume() > 0` pour une surface fermée).
- Formats : NIfTI en entrée, GIfTI pour surfaces et champs par sommet, VTK `.vtu` côté gyroxide.
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
multi-BOUNTI-43 (98,4 %) ; HD95 médian 0,17 et 0,19 mm ; épaisseur corticale médiane 1,66 et 1,47 mm.
Comparaison à fetpype (19 sessions) : blanche conforme 46/46 contre 24/38, HD95 0,18 contre 0,40 mm.
Présentation : https://claude.ai/artifact/DVcshJu5TQaxDFMLKSLWdZ
Reste en P0 : 7 hémisphères dont la piale ne démarre pas (plis résiduels de la blanche après 28 SA),
2 sous le seuil de distance ; sommets de piale retenus dans les sillons étroits (5 à 10 % après 30 SA) ;
chargement effectif dans gyroxide.

P1 : intégrateur ODE dans Burn, réseaux de déformation à la CoSeg (faiblement supervisés par les
segmentations de medoxide), opérateurs différentiels, métriques (épaisseur, profondeur sulcale,
gyrification, métrique spectrale), projection des lobes multi-BOUNTI, lecture `.vtu` de gyroxide.

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
