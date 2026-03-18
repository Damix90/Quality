import numpy as np
import matplotlib.pyplot as plt

# =========================
# Simulation de la réalité
# =========================
np.random.seed(42)

dt = 1.0
n_steps = 50

# Position réelle du drone
true_position = []
x_true = 0.0
velocity_true = 1.0  # 1 m/s

for _ in range(n_steps):
    x_true += velocity_true * dt
    true_position.append(x_true)

true_position = np.array(true_position)

# Mesures GPS bruitées
gps_noise_std = 3.0
gps_measurements = true_position + np.random.normal(0, gps_noise_std, n_steps)

# =========================
# Filtre de Kalman 1D
# Etat = [position, vitesse]
# =========================

# Etat initial estimé
x = np.array([[0.0],   # position estimée
              [0.0]])  # vitesse estimée

# Matrice de transition
F = np.array([[1, dt],
              [0, 1]])

# Matrice d'observation
H = np.array([[1, 0]])  # on observe seulement la position

# Covariance initiale de l'incertitude
P = np.array([[1000, 0],
              [0, 1000]], dtype=float)

# Bruit du modèle
q_position = 0.1
q_velocity = 0.1
Q = np.array([[q_position, 0],
              [0, q_velocity]])

# Bruit de mesure GPS
R = np.array([[gps_noise_std**2]])

estimated_positions = []
estimated_velocities = []

for z in gps_measurements:
    # =====================
    # 1. Prédiction
    # =====================
    x = F @ x
    P = F @ P @ F.T + Q

    # =====================
    # 2. Mise à jour
    # =====================
    z = np.array([[z]])
    y = z - (H @ x)                    # innovation
    S = H @ P @ H.T + R                # covariance innovation
    K = P @ H.T @ np.linalg.inv(S)     # gain de Kalman

    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    estimated_positions.append(x[0, 0])
    estimated_velocities.append(x[1, 0])

# =========================
# Affichage
# =========================
plt.figure(figsize=(10, 6))
plt.plot(true_position, label="Position réelle")
plt.scatter(range(n_steps), gps_measurements, label="Mesure GPS bruitée", s=25)
plt.plot(estimated_positions, label="Position estimée (Kalman)")
plt.xlabel("Temps")
plt.ylabel("Position")
plt.title("Fusion de données GPS + modèle de mouvement avec filtre de Kalman")
plt.legend()
plt.grid(True)
plt.show()
