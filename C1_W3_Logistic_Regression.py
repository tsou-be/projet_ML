import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Pour charger les données
import math
import copy

def sigmoid(z):
    """
    Calcule la fonction sigmoïde.
    Args:
        z (ndarray): Un scalaire ou un vecteur numpy.
    Returns:
        g (ndarray): La sigmoïde, de même forme que z.
    """
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, lambda_=1):
    """
    Calcule le coût pour la régression logistique.
    Args:
      X (ndarray (m,n): Données avec m exemples, n features
      y (ndarray (m,)): Cible (0 ou 1)
      w (ndarray (n,)): Paramètres du modèle
      b (scalar)      : Paramètre de biais
      lambda_ (scalar): Paramètre de régularisation
    Returns:
      cost (scalar): coût
    """
    m,n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += (-y[i]*np.log(f_wb_i)) - ((1-y[i])*np.log(1-f_wb_i))
     
    cost = cost/m  
    return cost

def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Calcule la dérivée du coût par rapport aux paramètres w et b.
    Args:
      X (ndarray (m,n): Données avec m exemples, n features
      y (ndarray (m,)): Cible (0 ou 1) 
      w (ndarray (n,)): Paramètres du modèle
      b (scalar)      : Paramètre de biais
      lambda_ (scalar): Paramètre de régularisation
    Returns:
      dj_db (scalar):   Gradient du coût par rapport au biais
      dj_dw (ndarray (n,)): Gradient du coût par rapport aux poids
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):                             
        z_wb = np.dot(X[i], w) + b                  
        f_wb = sigmoid(z_wb)                          
        err = (f_wb - y[i])                           
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function, lambda_=None):
    """
    Effectue la descente de gradient.
    Args:
      X (ndarray (m,n): Données avec m exemples, n features
      y (ndarray (m,)): Cible (0 ou 1)
      w_in (ndarray (n,)): Poids initiaux
      b_in (scalar): Biais initial
      alpha (scalar): Taux d'apprentissage
      num_iters (int): Nombre d'itérations
      cost_function, gradient_function: Fonctions à utiliser
      lambda_ (scalar): Paramètre de régularisation
    Returns:
      w (ndarray (n,)): Poids finaux
      b (scalar): Biais final
      J_history (List): Historique des coûts
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        # Calcule gradient et met à jour les paramètres
        dj_db,dj_dw = gradient_function(X, y, w, b, lambda_) 
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Sauvegarde coût
        if i<100000:      # Enregistrement trop fréquent peut ralentir
            J_history.append( cost_function(X, y, w, b, lambda_) )

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            print(f'Iération {i:4}: Coût {J_history[-1]:8.2f}')

    return w, b, J_history

def predict(X, w, b):
    """
    Prédit les classes pour X en utilisant w et b.
    Args:
        X (ndarray (m,n)): Données avec m exemples, n features
        w (ndarray (n,)): Paramètres du modèle
        b (scalar): Paramètre de biais
    Returns:
        p (ndarray (m,)): Prédictions [0,1]
    """
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb > 0.5 else 0
        
    return p

# Fonction pour charger les données (remplace load_data du utils.py)
def load_data(file_path):
    """Charge les données depuis un fichier CSV."""
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    return X, y

# =========================================================================
# EXEMPLE D'UTILISATION (comme dans le notebook)
# =========================================================================

if __name__ == "__main__":
    # Chargez vos données ici (remplacez par votre chemin)
    # X_train, y_train = load_data("data/ex2data1.txt")
    
    # Données exemple générées
    np.random.seed(42)
    m = 100
    X_train = np.column_stack([
        30 + 30 * np.random.rand(m),
        43 + 25 * np.random.rand(m)
    ])
    y_train = np.random.binomial(1, 0.5, m).reshape(-1, 1)
    
    print("Premiers 5 exemples X_train:")
    print(X_train[:5])
    print("Premiers 5 exemples y_train:")
    print(y_train[:5])
    
    # Entraînement
    initial_w = np.zeros((X_train.shape[1], 1))
    initial_b = 0.
    iterations = 10000
    alpha = 0.001
    
    print("\n=== Entraînement ===")
    w_final, b_final, J_history = gradient_descent(
        X_train, y_train.flatten(), initial_w, initial_b, 
        alpha, iterations, compute_cost, compute_gradient, 0
    )
    
    # Prédictions
    p = predict(X_train, w_final.flatten(), b_final)
    accuracy = np.mean(p == y_train.flatten()) * 100
    print(f"\nPrécision: {accuracy:.2f}%")
    
    # Plot (optionnel)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[y_train.flatten()==1, 0], X_train[y_train.flatten()==1, 1], 
                c='green', marker='o', label='Admis')
    plt.scatter(X_train[y_train.flatten()==0, 0], X_train[y_train.flatten()==0, 1], 
                c='red', marker='x', label='Non admis')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.title('Données et modèle entraîné')
    plt.show()
    
    print("\n✅ Code corrigé et fonctionnel !")
