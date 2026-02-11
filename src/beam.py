"""
Beam and Material Classes for Modal Analysis

This module defines Material and Beam classes with presets for common engineering materials.
These classes provide a clean, object-oriented interface for defining structural properties.

Author: Kaan Gokbayrak, Purdue University
"""

from dataclasses import dataclass
from typing import ClassVar
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
class Beam:
    """
    Represents a rectangular cross-section beam with geometry and material properties.
    
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
    """
    material: Material
    length: float
    width: float
    thickness: float
    
    def __post_init__(self):
        """Validate beam geometry."""
        if self.length <= 0:
            raise ValueError(f"Length must be positive, got {self.length}")
        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")
        if self.thickness <= 0:
            raise ValueError(f"Thickness must be positive, got {self.thickness}")
    
    @property
    def area(self) -> float:
        """
        Calculate cross-sectional area.
        
        Returns
        -------
        float
            Cross-sectional area [m²]
        """
        return self.width * self.thickness
    
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
        return (self.width * self.thickness**3) / 12.0
    
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
