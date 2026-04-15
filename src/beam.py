"""
Beam and Material Classes for Modal Analysis

This module defines Material and Beam classes with presets for common engineering materials.
These classes provide a clean, object-oriented interface for defining structural properties.

Author: Kaan Gokbayrak, Purdue University
"""

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Union
import numpy as np


@dataclass
class Material:
    """
    Represents an engineering material with mechanical properties.
    
    Attributes
    ----------
    name : str
        Material name
    E : float
        Young's modulus [Pa]
    rho : float
        Density [kg/m³]
    nu : float
        Poisson's ratio [-]
    """
    name: str
    E: float
    rho: float
    nu: float
    
    def __post_init__(self):
        """Validate material properties."""
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if self.rho <= 0:
            raise ValueError(f"Density must be positive, got {self.rho}")
        if not 0 <= self.nu < 0.5:
            raise ValueError(f"Poisson's ratio must be in [0, 0.5), got {self.nu}")
    
    @classmethod
    def steel(cls) -> 'Material':
        """
        Create a steel material with typical properties.
        
        Returns
        -------
        Material
            Steel with E=200 GPa, ρ=7850 kg/m³, ν=0.3
        """
        return cls(name="Steel", E=200e9, rho=7850, nu=0.3)
    
    @classmethod
    def aluminum(cls) -> 'Material':
        """
        Create an aluminum material with typical properties.
        
        Returns
        -------
        Material
            Aluminum with E=69 GPa, ρ=2700 kg/m³, ν=0.33
        """
        return cls(name="Aluminum", E=69e9, rho=2700, nu=0.33)
    
    @classmethod
    def titanium(cls) -> 'Material':
        """
        Create a titanium material with typical properties.
        
        Returns
        -------
        Material
            Titanium with E=116 GPa, ρ=4500 kg/m³, ν=0.34
        """
        return cls(name="Titanium", E=116e9, rho=4500, nu=0.34)
    
    @classmethod
    def carbon_fiber(cls) -> 'Material':
        """
        Create a carbon fiber composite material with typical properties.
        
        Returns
        -------
        Material
            Carbon fiber with E=181 GPa, ρ=1600 kg/m³, ν=0.28
        """
        return cls(name="Carbon Fiber", E=181e9, rho=1600, nu=0.28)
    
    def __repr__(self) -> str:
        """Return engineering summary of material properties."""
        return (f"Material({self.name}: E={self.E/1e9:.1f} GPa, "
                f"ρ={self.rho:.0f} kg/m³, ν={self.nu:.2f})")


@dataclass
class RectangularSection:
    """
    Solid rectangular cross-section.

    Attributes
    ----------
    width : float
        Section width [m]
    thickness : float
        Section thickness (bending dimension) [m]
    """
    width: float
    thickness: float

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return self.width * self.thickness

    @property
    def I(self) -> float:
        """Second moment of area about the centroidal axis [m⁴]."""
        return self.width * self.thickness**3 / 12


@dataclass
class IBeamSection:
    """
    I-beam (wide-flange) cross-section.

    Attributes
    ----------
    flange_width : float
        Width of each flange [m]
    flange_thickness : float
        Thickness of each flange [m]
    web_height : float
        Clear height of the web (between flanges) [m]
    web_thickness : float
        Thickness of the web [m]
    """
    flange_width: float
    flange_thickness: float
    web_height: float
    web_thickness: float

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return 2 * self.flange_width * self.flange_thickness + self.web_height * self.web_thickness

    @property
    def I(self) -> float:
        """Second moment of area about the centroidal axis [m⁴]."""
        H = 2 * self.flange_thickness + self.web_height
        return (self.flange_width * H**3 - (self.flange_width - self.web_thickness) * (H - 2 * self.flange_thickness)**3) / 12


@dataclass
class HollowRectSection:
    """
    Hollow rectangular (box) cross-section.

    Attributes
    ----------
    outer_width : float
        Outer width [m]
    outer_thickness : float
        Outer thickness [m]
    inner_width : float
        Inner (void) width [m]
    inner_thickness : float
        Inner (void) thickness [m]
    """
    outer_width: float
    outer_thickness: float
    inner_width: float
    inner_thickness: float

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return self.outer_width * self.outer_thickness - self.inner_width * self.inner_thickness

    @property
    def I(self) -> float:
        """Second moment of area about the centroidal axis [m⁴]."""
        return (self.outer_width * self.outer_thickness**3 - self.inner_width * self.inner_thickness**3) / 12


@dataclass
class CircularSection:
    """
    Solid circular cross-section.

    Attributes
    ----------
    radius : float
        Radius of the circle [m]
    """
    radius: float

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return np.pi * self.radius**2

    @property
    def I(self) -> float:
        """Second moment of area about the centroidal axis [m⁴]."""
        return np.pi * self.radius**4 / 4


@dataclass
class HollowCircularSection:
    """
    Hollow circular (annular) cross-section.

    Attributes
    ----------
    outer_radius : float
        Outer radius [m]
    inner_radius : float
        Inner (void) radius [m]
    """
    outer_radius: float
    inner_radius: float

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)

    @property
    def I(self) -> float:
        """Second moment of area about the centroidal axis [m⁴]."""
        return np.pi * (self.outer_radius**4 - self.inner_radius**4) / 4


@dataclass
class TSection:
    """
    T-shaped cross-section (bending about the centroidal axis).

    Attributes
    ----------
    flange_width : float
        Width of the flange [m]
    flange_thickness : float
        Thickness of the flange [m]
    web_height : float
        Height of the web (below the flange) [m]
    web_thickness : float
        Thickness of the web [m]
    """
    flange_width: float
    flange_thickness: float
    web_height: float
    web_thickness: float

    @property
    def area(self) -> float:
        """Cross-sectional area [m²]."""
        return self.flange_width * self.flange_thickness + self.web_height * self.web_thickness

    @property
    def I(self) -> float:
        """Second moment of area about the centroidal axis [m⁴]."""
        total_area = self.area
        y_c = (
            self.flange_width * self.flange_thickness * (self.web_height + self.flange_thickness / 2)
            + self.web_height * self.web_thickness * (self.web_height / 2)
        ) / total_area
        I_flange = (
            self.flange_width * self.flange_thickness**3 / 12
            + self.flange_width * self.flange_thickness * (self.web_height + self.flange_thickness / 2 - y_c)**2
        )
        I_web = (
            self.web_thickness * self.web_height**3 / 12
            + self.web_thickness * self.web_height * (self.web_height / 2 - y_c)**2
        )
        return I_flange + I_web


SectionType = Union[RectangularSection, IBeamSection, HollowRectSection, CircularSection, HollowCircularSection, TSection]


@dataclass
class Beam:
    """
    Represents a beam with geometry, material, and optional cross-section properties.

    Attributes
    ----------
    material : Material
        Material properties
    length : float
        Beam length [m]
    width : float
        Beam width (in-plane dimension) [m]
    thickness : float
        Beam thickness (out-of-plane dimension, bending about neutral axis) [m]
    section : SectionType, optional
        Cross-section object. Defaults to ``RectangularSection(width, thickness)``
        when not provided.
    winkler_stiffness : float
        Distributed elastic foundation stiffness (Winkler model) [N/m²]. Default 0.
    """
    material: Material
    length: float
    width: float
    thickness: float
    section: Optional[SectionType] = field(default=None, repr=False)
    winkler_stiffness: float = 0.0
    
    def __post_init__(self):
        """Validate beam geometry and initialise default cross-section."""
        if self.length <= 0:
            raise ValueError(f"Length must be positive, got {self.length}")
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.thickness <= 0:
            raise ValueError(f"Thickness must be positive, got {self.thickness}")
        if self.section is None:
            self.section = RectangularSection(self.width, self.thickness)
        if self.winkler_stiffness < 0:
            raise ValueError(f"Winkler stiffness must be non-negative, got {self.winkler_stiffness}")
    
    @property
    def area(self) -> float:
        """
        Calculate cross-sectional area.

        Returns
        -------
        float
            Cross-sectional area [m²]
        """
        if isinstance(self.section, RectangularSection):
            return self.width * self.thickness
        return self.section.area

    @property
    def I(self) -> float:
        """
        Calculate second moment of area about the neutral axis.

        For a rectangular cross-section bending about the centroidal axis parallel
        to the width, the second moment is I = (width * thickness³) / 12.

        Returns
        -------
        float
            Second moment of area [m⁴]
        """
        if isinstance(self.section, RectangularSection):
            return (self.width * self.thickness**3) / 12.0
        return self.section.I
    
    @property
    def mass_per_length(self) -> float:
        """
        Calculate mass per unit length.
        
        Returns
        -------
        float
            Mass per unit length [kg/m]
        """
        return self.material.rho * self.area
    
    @property
    def total_mass(self) -> float:
        """
        Calculate total beam mass.
        
        Returns
        -------
        float
            Total mass [kg]
        """
        return self.mass_per_length * self.length
    
    def __repr__(self) -> str:
        """Return engineering summary of beam properties."""
        return (f"Beam({self.material.name}, L={self.length*1000:.1f} mm, "
                f"b={self.width*1000:.1f} mm, h={self.thickness*1000:.1f} mm, "
                f"m={self.total_mass*1000:.2f} g)")
