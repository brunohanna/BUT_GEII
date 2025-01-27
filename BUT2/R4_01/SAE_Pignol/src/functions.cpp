#include "functions.h"

// Déclaration des utilisateurs
User users[] = {
    {"pignol", "geii"}, // Utilisateur 'pignol'
    {"bruno", "geii"}   // Utilisateur 'bruno'
};
int numUsers = sizeof(users) / sizeof(users[0]);

// Fonction de hachage simple (à utiliser avec prudence)
String simpleHash(String input) {
    int hash = 0;
    for (unsigned int i = 0; i < input.length(); i++) {
        hash += input[i];
    }
    return String(hash);
}

// Vérifie si un utilisateur et un mot de passe sont corrects
String checkUser(String username, String password) {
    username.toLowerCase(); // Convertit le nom d'utilisateur en minuscules
    String passwordToCheck = useHashing ? simpleHash(password) : password;
    for (int i = 0; i < numUsers; i++) {
        if (users[i].username == username) {
            if (users[i].passwordHash == passwordToCheck) {
                return "OK";
            } else {
                return "Mot de passe faux";
            }
        }
    }
    return "Nom d'utilisateur non existant";
}

// Configure le serveur web
void configureServer(AsyncWebServer& server) {
    // Gestionnaire pour la route de connexion
    server.on("/login", HTTP_GET, [](AsyncWebServerRequest *request) {
        if (request->hasParam("username") && request->hasParam("password")) {
            String username = request->getParam("username")->value();
            String password = request->getParam("password")->value();
            String result = checkUser(username, password);
            request->send(200, "text/plain", result);
        } else {
            request->send(400, "text/plain", "Paramètres manquants");
        }
    });

    // Gestionnaire pour la route de réglage de la PWM
    server.on("/setPWM", HTTP_GET, [](AsyncWebServerRequest *request) {
        if (request->hasParam("value")) {
            int value = request->getParam("value")->value().toInt();
            setPWM( value); // Régle la PWM sur GPIO 15
            request->send(200, "text/plain", "PWM mis à jour");
        } else {
            request->send(400, "text/plain", "Paramètre manquant");
        }
    });

    // Ajoutez d'autres gestionnaires si nécessaire
}

// Fonction pour régler la PWM (peut être utilisée directement ou via le serveur web)
void setPWM(int value) {
    analogWrite(15, value); // Régle la PWM sur GPIO 15
}
