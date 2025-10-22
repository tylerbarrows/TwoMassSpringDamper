# Modeling Two Masses Connected via a Spring–Damper System

This project models the dynamics of **two masses connected by a spring and a damper**, with **frictional forces** acting on both masses. The system is analyzed and simulated using several different numerical and analytical approaches.

A motivating application is the **coupling interaction between train cars**, where the connection between cars (the coupling) can be modeled as a spring–damper system. The focus is on understanding the effects of different modeling methods on system behavior, energy conservation, and energy dissipation in the damper during an impact.

---

## System Overview

The system consists of:

* Two masses, ( m_1 ) and ( m_2 )
* A spring with stiffness ( k12 )
* A damper with damping coefficient ( c12 )
* Friction acting on both masses (f1 and f2)

I use the state space approach in building my equations starting with free body diagrams on both masses. The model is fairly general but I consider cases where there is a forcing function on one mass and this mass is at rest at t = 0. The other mass has no forcing function but has an initial velocity moving torwards the other mass. I also consider geometry constraints in some cases where the coupling (spring and damper) between the masses can only extend and compress so much. I assume perfectly elastic collisions (conservation of momentum and energy) to model that limit. 

---

## Modeling Approaches

Five methods are implemented and compared:

1. **Explicit Time-Stepping**
   Standard numerical integration using a small time step (e.g., explicit Euler).

2. **Dual Time Stepping**
   Uses an inner pseudo-time iteration to converge the solution at each real time step, improving stability and accuracy.

3. **Analytical (Integral Approximated via Single Riemann Sum)**
   Simplifies the analytical solution by assuming the integral term can be approximated with one Riemann sum.

4. **Analytical (Inverse Solution using SVD)**
   Solves the analytical system by computing the inverse through **Singular Value Decomposition (SVD)**.

5. **Analytical (Padé Approximation)**
   Uses a Padé rational approximation to approximate the system’s exponential response.

---

## Analysis Performed

After computing system responses using the five methods, I also looked at the following (only using the dual time stepping method). 

* **Geomtry Constraints:**
  I look at how the masses respond when I introduce the geometry constraint (limit how much coupling can expand and compress) and compare this response to the one without the constraint. 

* **Energy Conservation:**
  Check numerical consistency of total energy (kineti, potential, dissipated (friction and damper), input, and initial).

* **Force and Energy Dissipation in the Damper:**
  Analyzed as a function of the **initial velocity** of one mass (while the other mass starts at rest).

These analyses help characterize how the damper behaves under different impact velocities — relevant for applications like **train coupling** dynamics.

---

## Repository Structure

```
📦 two-mass-spring-damper
├── model.py                   # Main script: runs all analyses and plots results
├── solvingMethods.py          # Contains forcing input and five modeling methods
├── README.md                  # Project description (this file)
```

---

## Installation

1. Clone the respository:
git clone https://github.com/tylerbarrows/CFD.git
cd CFD

2. Install required dependencies
Numpy
Math
Matplotlib
Scipy

## Running the model
To run the model, run model.py. In model.py, one can change parameters (spring constant, mass, etc.).
