# surfoxide

Boîte à outils en Rust & [Burn](https://burn.dev) pour les surfaces corticales :
extraction des surfaces blanche (WM/GM) et piale (GM/CSF) à partir de segmentations,
contrôle de leur validité et mesures corticales. Chaque méthode est un module,
exposé comme une sous-commande de `surfx`.

surfoxide fait le lien, par les données, entre deux projets :

- [medoxide](https://github.com/rousseau/medoxide) produit les images reconstruites et les
  segmentations ((multi-)BOUNTI) au format NIfTI ;
- gyroxide simule le plissement cortical à partir des surfaces (GIfTI) et renvoie ses
  surfaces simulées, mesurées par le même code que les surfaces réelles.

**Périmètre : surfoxide ne traite que des surfaces.** Il ne produit ni ne lit de maillage
volumique : la tétraédrisation des surfaces qu'il livre est faite par gyroxide. Ce qui circule
entre les deux dépôts, ce sont des surfaces triangulaires (et des champs par sommet) respectant
le contrat ci-dessous.

| Sous-commande | Rôle |
|---|---|
| `surfx device` | Vérifie qu'un backend Burn calcule sur la machine (CPU, CUDA ou wgpu) |
| `surfx selftest` | Construit une icosphère et un tore et affiche leur bilan topologique |
| `surfx run` | Surfaces blanche et piale de chaque hémisphère, épaisseur corticale et masque du cortex (`.shape.gii`), rapport de contrat JSON |
| `surfx white` | Surface blanche de genre 0 de chaque hémisphère (GIfTI) : masque, correction topologique, marching cubes, Taubin, remaillage isotrope, réparation ; rapport de contrat JSON |
| `surfx eval` | Distance d'une surface à la frontière de la segmentation (Hausdorff 95 %), qualité du maillage, contrat |
| `surfx check` | Contrôle du contrat : topologie, orientation, auto-intersections ; imbrication et épaisseur WM/piale ; rapport JSON |
| `surfx mask` | Masque blanc fermé d'un hémisphère à partir d'une segmentation BOUNTI-19 ou multi-BOUNTI-43 |
| `surfx convert` | Convertit une surface (GIfTI, VTP, PLY, STL, FreeSurfer) et affiche son bilan topologique |

## Démarrage rapide

Prérequis : [Rust](https://rustup.rs) 1.95 ou plus récent. Pour le GPU NVIDIA : pilote et
CUDA Toolkit (voir `scripts/setup-linux.sh` pour Ubuntu, `scripts/setup-wsl.sh` pour WSL).

```bash
cargo test --release
cargo run --release -- selftest
cargo run --release -- run sub-01_dseg.nii.gz -o out/sub-01       # WM et piale par hémisphère
cargo run --release -- device                                   # CPU
cargo run --release --features cuda -- device --backend cuda    # GPU NVIDIA
```

La première compilation avec `--features cuda` télécharge un paquet LLVM d'environ 45 Mo
(`~/.cache/tracel/`), utilisé par CubeCL.

## Structure

```
crates/
  surfoxide-core/   maillage, volume avec affine, adjacence, topologie, surfaces de référence ; sans Burn
  surfoxide-burn/   maillages et champs ↔ tenseurs Burn
  surfoxide-io/     NIfTI, GIfTI, VTP, PLY, STL, FreeSurfer
  surfoxide-mask/   tables de labels (multi-)BOUNTI, masques d'hémisphère
  surfoxide-topofix/ correction topologique voxelique (Bazin, points simples 26/6)
  surfoxide-extract/ marching cubes à connexité cohérente, lissage de Taubin
  surfoxide-check/  BVH, intersections exactes, distances, rapport de contrat
  surfoxide-remesh/ remaillage isotrope (Botsch–Kobbelt) sur une table des coins modifiable
  surfoxide-repair/ suppression des auto-intersections par déplacement de sommets
  surfoxide-pial/   piale par les lignes de courant du champ de Laplace, sillons enfouis
  surfoxide-metrics/ Laplace–Beltrami éléments finis, courbure (Rusinkiewicz), enveloppe convexe,
                    profondeur sulcale DPF* invariante d'échelle (Dieudonné, Auzias, Lefèvre 2026)
  surfx/            binaire CLI : une sous-commande par module
scripts/
  setup-linux.sh    préparation d'une machine Ubuntu avec GPU NVIDIA
  fetch-data.sh     rapatrie les segmentations (et T2w) depuis Garage dans $SURFOXIDE_DATA
  validate.py       validation sur la cohorte et comparaison à fetpype
  illustrate.py     figure T2w, segmentation et surfaces pour quelques âges (numpy, nibabel, matplotlib)
  plate.py          planche de toutes les surfaces d'un dossier, de la plus jeune à la plus âgée
  setup-wsl.sh      préparation d'une machine WSL avec GPU NVIDIA
```

## Conventions de données

- Coordonnées dans le repère monde RAS, en mm ; l'affine de l'image source est conservée.
- Faces orientées dans le sens trigonométrique vu de l'extérieur (normales sortantes).
- Surfaces et champs par sommet en GIfTI, images en NIfTI ; aucun maillage volumique (le `.vtu` reste
  côté gyroxide).
