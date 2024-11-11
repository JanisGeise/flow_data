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
- simulation end time $t_\mathrm{end}=t_\mathrm{conv}/xx = xxx$
- velocity, pressure, and force coefficients are sampled every $\Delta t_\mathrm{write} = t_\mathrm{conv}/xx = xxx$