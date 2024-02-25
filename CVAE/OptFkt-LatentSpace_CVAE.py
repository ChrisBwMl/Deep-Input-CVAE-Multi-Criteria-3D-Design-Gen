from scipy.optimize import minimize

def optimize_construction(latent_vector_0, decoder):
    """
    Optimiert den latenten Vektor, um den Abruf von Konstruktionsvorschlägen zu verbessern.
    
    Parameters:
    latent_vector_0 (numpy.ndarray): Initialer latenter Vektor.
    decoder (keras.Model): Decoder-Modell, das den latenten Vektor in die Ausgabe umwandelt.
    
    Returns:
    numpy.ndarray: Optimierter latenter Vektor.
    """
    def optimization_function(latent_vector, decoder):
        """
        Ziel-Funktion für die Optimierung.
        Setzt bestimmte Ausgangswerte auf 0 oder 1 und minimiert die Eindeutigkeit der Konstruktionsvorschläge.
        
        Parameters:
        latent_vector (numpy.ndarray): Der zu optimierende latente Vektor.
        decoder (keras.Model): Decoder-Modell.
        
        Returns:
        int: Anzahl der eindeutigen Konstruktionsvorschläge nach der Optimierung.
        """
        output = decoder.predict(latent_vector.reshape(1, -1))
        output[output > 0.1] = 1
        output[output < 0.1] = 0
        unambiguous = np.sum(output != 0)
        return -unambiguous  # Negative, da minimize die Funktion minimieren soll

    result = minimize(fun=optimization_function, x0=latent_vector_0, args=(decoder), method='BFGS')
    latent_vector_optimal = result.x

    return latent_vector_optimal

# Beispielaufruf
latent_vector_initial = np.zeros((1, 274))  # Initialer latenter Vektor mit Nullen
optimal_latent_vector = optimize_construction(latent_vector_initial, decoder)
print("Optimaler Latenter Vektor:", optimal_latent_vector)
