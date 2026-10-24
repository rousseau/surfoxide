//! `surfoxide-learn` : réseaux de déformation de surfaces corticales (P1), à la CortexODE.
//!
//! Un encodeur convolutif 3D lit les classes de tissu du cadrage de l'hémisphère ; un champ de vitesse (MLP sur
//! les caractéristiques échantillonnées aux sommets, à trois échelles, et sur le temps) déplace la surface
//! blanche le long d'une ODE intégrée par `surfoxide-ode` ; la perte compare la surface obtenue à la piale du
//! flot (même triangulation, supervision sommet à sommet) et à la frontière des tissus (distance signée).
//!
//! - [`data`] : jeu d'entraînement écrit par `surfx dataset` / `scripts/dataset.py`.
//! - [`model`] : encodeur, champ de vitesse, déformation.
//! - [`train`] : boucle d'entraînement et évaluation.

pub mod data;
pub mod model;
pub mod train;
