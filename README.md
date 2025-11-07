# Common flow problems for modal analysis implemented in OpenFOAM

## Dependencies

- OpenFOAM-v2406

## Test cases

To run a simulation case, create a folder called *run*, create a copy of the simulation setup in the *run* folder, and execute the *Allrun* script.

### 2D laminar flow past a cylinder in a narrow channel at $Re=100$

The case setup folder is *cylinder_2D_Re100*. The main simulation parameters are:
- parabolic inlet velocity profile with $U_\mathrm{max} = 1.0m/s$
- $Re=U_\mathrm{max}d/\nu=1\cdot 0.1/10^{-3}=100$ ($d$ - cylinder diameter, $\nu$ - kinematic viscosity)
- convective time scale: $t_\mathrm{conv} = d/U_\mathrm{max} = 0.1/1 = 0.1s$
- simulation end time $t_\mathrm{end}=200t_\mathrm{conv} = 20s$
- velocity, pressure, and force coefficients are sampled every $\Delta t_\mathrm{write} = t_\mathrm{conv}/50 = 2\times10^{-3}s$

To run the simulation with a different mesh resolution, comment/uncomment the corresponding lines in the *Allrun* script
The notebook [cylinder_2D_Re100.ipynb](./cylinder_2D_Re100.ipynb) contains results for the mesh dependency of the force coefficients, the shear stress velocity at the cylinder's surface, and the singular value decomposition of the vorticity field.

## 3D turbulent flow at $Re=3900$

- uniform inlet velocity of $u_x = U_\mathrm{in} = 39m/s$
- $Re=U_\mathrm{max}d/\nu=39\cdot 0.1/10^{-3}=3900$ ($d$ - cylinder diameter, $\nu$ - kinematic viscosity)
- $t_\mathrm{conv} = d/U_\mathrm{in} = 0.1/39.0 \approx 2.5641\times 10^{-3}s$
- simulation end time $t_\mathrm{end}=3900t_\mathrm{conv} = 10s$
- snapshots with velocity and pressure are sampled every $\Delta t_\mathrm{write} = t_\mathrm{conv}/10.25 = 2.5\times10^{-4}s$
- force coefficients are sampled every $\Delta t_\mathrm{write} = t_\mathrm{conv}/51.2 = 1\times10^{-5}s$
- the transient phase is completed after $t_\mathrm{end}=75t_\mathrm{conv} = 0.19225s$
- `mean` and `prime2Mean` quantities are sampled every $\Delta t_\mathrm{write} = t_\mathrm{conv}/0.1025 = 2.5\times10^{-2}s$

## References
- 2D laminar flow past a cylinder in a narrow channel at $Re=100$:
  - Schäfer, M., Turek, S., Durst, F., et al.: “[Benchmark Computations of Laminar Flow
  Around a Cylinder](https://link.springer.com/chapter/10.1007/978-3-322-89849-4_39)”. In: Flow Simulation with High-Performance Computers II. Ed. by
  Hirschel, E. H. Vol. 48. Notes on Numerical Fluid Mechanics (NNFM). Wiesbaden: Vieweg+Teubner
  Verlag, 1996, pp. 547–566. doi: 10.1007/978-3-322-89849-4_39.
- 3D turbulent flow at $Re = 3900$:
  - Lehmkuhl O. Lehmkuhl, I. Rodrı́guez, R. Borrell, and A. Oliva. “[Low-frequency unsteadiness in the vortex formation
  region of a circular cylinder](https://pubs.aip.org/aip/pof/article/25/8/085109/102970/Low-frequency-unsteadiness-in-the-vortex-formation)”. In: Physics of Fluids 25.8 (2013), p. 085109. doi: 10.1063/1.4818641, 
