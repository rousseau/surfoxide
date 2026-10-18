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

| Sous-commande | Rôle |
|---|---|
| `surfx device` | Vérifie qu'un backend Burn calcule sur la machine (CPU, CUDA ou wgpu) |
| `surfx selftest` | Construit une icosphère et un tore et affiche leur bilan topologique |
| `surfx convert` | Convertit une surface (GIfTI, VTP, PLY, STL, FreeSurfer) et affiche son bilan topologique |

## Démarrage rapide

Prérequis : [Rust](https://rustup.rs) 1.95 ou plus récent. Pour le GPU NVIDIA : pilote et
CUDA Toolkit (voir `scripts/setup-linux.sh` pour Ubuntu, `scripts/setup-wsl.sh` pour WSL).

```bash
cargo test --release
cargo run --release -- selftest
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
  surfx/            binaire CLI : une sous-commande par module
scripts/
  setup-linux.sh    préparation d'une machine Ubuntu avec GPU NVIDIA
  setup-wsl.sh      préparation d'une machine WSL avec GPU NVIDIA
```

## Conventions de données

- Coordonnées dans le repère monde RAS, en mm ; l'affine de l'image source est conservée.
- Faces orientées dans le sens trigonométrique vu de l'extérieur (normales sortantes).
- Surfaces en GIfTI, maillages volumiques en VTK `.vtu`, images en NIfTI.
