%% Lab 3 SOM - Réseau SOM de Kohonen
clear; close all; clc;
%% Carte Linéaire

% Manipulation des données
S = load('samples1.mat');   % charge la structure
data = S.data;              % extrait la matrice 16×2


% Visualisation des données
figure;
scatter(data(:,1), data(:,2))
title('Données d\''entrée')
xlabel('X1')
ylabel('X2')
axis equal

% Taille du réseau
mapsize = [1 3];
alpha=geometric(1,.9,100); % taux d'apprentissage
%% Visualisation de l'évolution des poids
[c,p]=som(data,mapsize,'alpha',alpha,'protomap',1);

%%  Visualisation des prototypes

n = size(data,1);   % 16
m = size(p,1);      % 3 prototypes

% Couleur des données (rouge)
dataColor = repmat([1 0 0], n, 1);

% Couleur des prototypes (vert ou autre)
protoColor = repmat([0 1 0], m, 1);

% Matrice de couleurs complète
co = [dataColor ; protoColor];

scatter([data(:,1);p(:,1)], [data(:,2);p(:,2)], 50, co);
title('Données et prototypes')
xlabel('X1')
ylabel('X2')
axis equal

%%
for classe = 1:3
    fprintf("Classe %d :\n", classe);

    % Trouver les individus appartenant à cette classe
    index = find(c == classe);
    fprintf("Indices des individus : ");
    disp(index');

    % Calcul du centre de gravité des individus de la classe
    centroid = mean(data(index, :), 1);
    fprintf("Centre de gravité : [%f, %f]\n", centroid(1), centroid(2));

    % Prototype associé
    prototype = p(classe, :);
    fprintf("Prototype du SOM : [%f, %f]\n\n", prototype(1), prototype(2));
end

%% Conclusion (Question 4)
% En comparant les centres de gravité des classes avec les prototypes du SOM,
% on remarque qu'ils sont assez proches pour chaque classe. Cela montre que
% le réseau SOM a bien appris à représenter les groupes de données : chaque
% neurone correspond à un ensemble cohérent de points.
%
% Les prototypes ne coïncident pas exactement avec les centroïdes, ce qui est
% normal. Le SOM cherche aussi à préserver la topologie, c’est-à-dire à garder
% un certain ordre entre les neurones sur la carte linéaire. À cause de cette
% contrainte, les prototypes peuvent être légèrement déplacés par rapport aux
% centres de gravité exacts.
%
% En résumé :
% - Les prototypes représentent bien les classes apprises.
% - Les petites différences avec les centroïdes viennent de la préservation
%   de l’ordre/topologie imposée par la carte SOM.