import numpy as np
import matplotlib.pyplot as plt

# 1. Definicja zakresu temperatur
T = np.linspace(250, 400, 500)

size = 8

# 2. Funkcje limitów
Q_viscous = 5 * np.exp(0.06 * (T - 250))
Q_sonic = 60 + 1.5 * (T - 250)**0.85
Q_entrainment = 220 - 0.015 * (T - 340)**2
Q_capillary = 160 - 0.04 * (T - 250)**1.2
Q_boiling = 190 - 0.02 * (T - 250)**1.5

# 3. Wyznaczenie strefy bezpiecznej
Q_safe_zone = np.minimum.reduce([Q_viscous, Q_sonic, Q_entrainment, Q_capillary, Q_boiling])

# 4. Inicjalizacja wykresów
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300, sharey=True)

# --- LEWY WYKRES ---
ax1.plot(T, Q_viscous, label='Viscous', color='purple', linestyle='--', linewidth=1.5)
ax1.plot(T, Q_sonic, label='Sonic', color='blue', linestyle='-.', linewidth=1.5)
ax1.plot(T, Q_entrainment, label='Entrainment', color='green', linestyle=':', linewidth=1.5)
ax1.plot(T, Q_capillary, label='Capillary', color='red', linewidth=1.5)
ax1.plot(T, Q_boiling, label='Boiling', color='orange', linewidth=1.5)

ax1.set_xlabel('Operating Temperature (T), K', fontsize=size)
ax1.set_ylabel('Heat Transport Capacity (Q), W', fontsize=size)
ax1.set_xlim(250, 400)
ax1.set_ylim(0, 250)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=size, framealpha=1.0)

# KRYTYCZNA ZMIANA: Twarde wymuszenie rozmiaru liczb na osi X i Y
ax1.tick_params(axis='both', which='major', labelsize=size) 

# --- PRAWY WYKRES ---
ax2.plot(T, Q_safe_zone, color='black', linewidth=1.5, label='Limiting Boundary')
ax2.fill_between(T, 0, Q_safe_zone, color='gray', alpha=0.2, label='Safe Zone')

ax2.set_xlabel('Operating Temperature (T), K', fontsize=size)
ax2.set_xlim(250, 400)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right', fontsize=size, framealpha=1.0)

# KRYTYCZNA ZMIANA: Twarde wymuszenie rozmiaru liczb na osi X i Y
ax2.tick_params(axis='both', which='major', labelsize=size) 

# 5. Formatowanie końcowe
plt.tight_layout()
plt.show()