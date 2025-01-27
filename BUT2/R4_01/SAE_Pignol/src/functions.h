// functions.h
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Arduino.h>
#include <ESPAsyncWebServer.h>
#include <FS.h>
#include <ESP8266WiFi.h>
#include <ESPAsyncWebServer.h>

extern bool useHashing; // Contrôle du hachage

typedef struct {
  String username;
  String passwordHash;
} User;

void setPWM(int value);
void configureServer(AsyncWebServer& server);
String checkUser(String username, String password);
String simpleHash(String input);

#endif