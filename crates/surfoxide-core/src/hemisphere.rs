//! Hémisphère cérébral.

use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hemisphere {
    Left,
    Right,
}

impl Hemisphere {
    pub const BOTH: [Hemisphere; 2] = [Hemisphere::Left, Hemisphere::Right];

    /// Entité BIDS `hemi-L` / `hemi-R`.
    pub fn bids(self) -> &'static str {
        match self {
            Hemisphere::Left => "L",
            Hemisphere::Right => "R",
        }
    }

    pub fn is_left(self) -> bool {
        self == Hemisphere::Left
    }
}

impl fmt::Display for Hemisphere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Hemisphere::Left => "gauche",
            Hemisphere::Right => "droit",
        })
    }
}

impl FromStr for Hemisphere {
    type Err = String;

    /// Accepte `L`, `left`, `gauche`, `R`, `right`, `droit` (sans tenir compte de la casse).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "l" | "left" | "gauche" | "lh" => Ok(Hemisphere::Left),
            "r" | "right" | "droit" | "rh" => Ok(Hemisphere::Right),
            _ => Err(format!("hémisphère inconnu « {s} » (L ou R attendu)")),
        }
    }
}
