function [CLASSES, proto, pMap, rMap] = som (data, mapsize, varargin)

% SOM
%          [classes, proto, pMap, rMap] = som (data, mapsize)
%
% Classification automatique avec preservation topologique sur une
% carte lineaire ou carree.
%
% Variables d'entree :
%
%   data    :  matrice de vecteurs-lignes (valeurs reelles)
%   mapsize :  vecteur-ligne indiquant le nombre de lignes et colonnes
%              de la carte de Kohonen (max : 2D)
%
% Variables de sortie :
%
%   classes :  classes d'appartenance des donnees (vecteur-colonne)
%   proto   :  prototypes des classes (matrice de vecteurs-lignes)
%   pMap    :  coordonnees des prototypes sur la carte de Kohonen (matrice)
%   rMap    :  cartes de reponse des entrees (liste de vecteurs ou matrices)
%
%
% Parametres d'entree optionnels :
% --------------------------------
%
% Ils s'ajoutent a la suite de data et mapsize, sous forme de liste
% 'nom', valeur. Exemple :
%
%        	w = som (data, mapsize, 'nom1', valeur1, 'nom2', valeur2)
%
% 'alpha'    :  fonction de decroissance du taux d'apprentissage (vecteur).
%               Le vecteur alpha doit expliciter les valeurs individuelles
%               de alpha(t) pour chaque epoque.
%               Defaut :  decroissance lineaire de alpha0 a 0
%
% 'alpha0'   :  taux d'apprentissage initial (scalaire).  Un seul de
%               alpha0 ou alpha devrait etre fourni.
%               Defaut :  alpha0 = 1.
%
% 'radius'   :  fonction de decroissance du voisinage (vecteur). Ce vecteur
%               doit expliciter les valeurs individuelles de radius(t) pour
%               chaque epoque.
%               Defaut :  decroissance lineaire de radius0 a 0
%
% 'radius0'  :  rayon initial du voisinage (scalaire).  On pose que la
%               distance entre deux neurones voisins (situes sur la meme
%               coordonnees x ou y) est de 1.  Un seul de radius0 ou radius
%               devrait etre fourni.
%               Defaut :  radius0 = 0.5 * (nombre de neurones sur la
%                                          plus grande dimension )
%
% 'sigma'    :  etalement horizontal de la courbe de ponderation radiale
%               (vecteur).
%               Defaut : sigma constant de 2
%
% 'epochs'   :  nombre d'epoques (scalaire).
%               Defaut :  epochs = 500 * mapsize
%               Si alpha, radius ou sigma sont fournis par l'utilisateur,
%               epochs = minimum de temps defini par ces fonctions
%
% 'protomap' :  affiche le deploiement des prototypes durant l'apprentissage
%               si protomap = 1.  La carte est visualisee en fonction des 
%               deux premiers parametres des donnees.
%               Defaut :  protomap = 0
%
% 'datamap'  :  affiche la reponse des neurones de sortie pour chaque
%               individu apres l'apprentissage, si datamap = 1
%               Defaut :  datamap = 0
%
% 'datatags' :  affiche le nom des donnees (en liste de string) en haut des
%               cartes de reponse.  Cette option n'est pertinente que si 
%               datamap = 1
%

% Appels :  aucun

% Ameliorations possibles :
%
% - ameliorer le mecanisme d'etiquettage.  Pour l'instant, la classe 
%   d'appartenance correspond au seul neurone gagnant ; critere simple,
%   mais aussi reducteur.  Selon ce critere, deux individus tres voisins
%   appartiendront a deux classes differentes parce qu'ils n'activent pas
%   exactement le meme neurone sur la carte.  La logique voudrait qu'une
%   classe soit definie sur une certaine etendue, et non pas sur un seul
%   neurone.  Avenues possibles :  rayon a fournir par l'utilisateur,
%   seuillage de regions.
%
%
% - etalement horizontal de la courbe de ponderation radiale : sigma
%
%   Version 1 : sigma fixe (=1).  Resultat acceptable, mais il n'est pas
%   logique que la courbe de ponderation ait toujours la meme ouverture.
%   Au-dela d'un certain rayon (environ 2 pour sigma =1), la correction
%   de poids devient tres faible, voire nulle.  Pour une grande carte,
%   ou la correction doit s'etendre sur un plus grand rayon, la correction
%   ne se fait pas alors.
%
%   Version 2 : sigma fixe, mais calcule en fonction de radius0.  On voit 
%   bien la forme des gaussiennes sur la carte si sigma est petit, mais la
%   zone active est trop large : trop de neurones voisins ont une activite
%   comparable. Il n'y a pas de raffinement local de l'apprentissage.
%   Si sigma est trop grand, la carte est morcelee en zones coupees au
%   couteau.  Bref, pas tres bon a date ; la version 1 etait preferable.
%
%   Version 3 : sigma variable, pour retrecir la zone gagnante. Le sigma
%   pourrait etre fonction du rayon courant, ou non.  Resultat :
%   semblable a la version 2
%
%   Conclusion :  pas de sigma optimal ?
%
%
% - ajouter un parametre d'entree weights.  Ceci servirait surtout a poursuivre
%   un apprentissage anterieur sans tout recommencer depuis le debut.  Pourrait
%   etre utile aussi pour initialiser les poids plus au centre, ou a n'importe
%   quel endroit juge plus approprie, ou a classifier des individus qui ne par-
%   ticipent pas a l'apprentissage


% Auteur :  Normand Gregoire 
% Cours  :  GPA-779, lab
% Date   :  6 decembre 1999


global flagsOfComputedDistanceMaps ;
global minResponse maxResponse ;
global protomapRequested stopEverything ;
global dataVarName ;

TRUE  = 1 ;
FALSE = 0 ;

stopEverything = FALSE ;

CLASSES = [] ;
proto   = [] ;
pMap    = [] ;


%============================ Securite =================================

if nargin < 2,
   fprintf (1, '\nNombre insuffisant d''arguments en entree\n') ;
   return ;
end

if length(mapsize) ~= 2 | any(mapsize <= 0) | size(mapsize,1) ~= 1,
   fprintf (1,'\nLes dimensions de la carte sont mal definies\n') ;
   return ;
end

% Pour l'affichage des cartes de reponse

dataVarName = inputname(1) ;

if isempty(dataVarName),
   dataVarName = 'data' ;
end

% Valeurs par defaut des parametres optionnels

protomapRequested = FALSE ;
datamapRequested  = FALSE ;
alpha0  = 1 ;
radius0 = 0.5 * max(mapsize) ;


% Identification des parametres optionnels fournis en entree

if ~isempty(varargin)

	msg = sprintf ('%c%s%c%s%c', char(10), ...
			'Les arguments d''entree optionnels doivent etre specifies par paires :', ...
			 char(10), 'nom d''argument, valeur, nom d''argument, valeur, etc', ...
			 char(10) ) ;

	if rem(length(varargin),2) ~= 0				% nombre pair d'arguments ?
		fprintf (1,'%s', msg) ;
		return ;
	end

	for i = 1:2:length(varargin),

		% Validite de la paire

		if ~iscellstr(varargin(i)),
			fprintf (1,'%s', msg) ;
			return ;
		end

		% Saisie des valeurs

		if     strcmpi(varargin{i},'epochs'  )  epochs  = varargin{i+1} ;
		elseif strcmpi(varargin{i},'alpha0'  )  alpha0  = varargin{i+1} ;
		elseif strcmpi(varargin{i},'alpha'   )  alpha   = varargin{i+1} ;
		elseif strcmpi(varargin{i},'radius0' )  radius0 = varargin{i+1} ;
		elseif strcmpi(varargin{i},'radius'  )  radius  = varargin{i+1} ;
		elseif strcmpi(varargin{i},'sigma'   )  sigma   = varargin{i+1} ;
      elseif strcmpi(varargin{i},'protomap')  protomapRequested = varargin{i+1} ;
      elseif strcmpi(varargin{i},'datamap' )  datamapRequested  = varargin{i+1} ;
      elseif strcmpi(varargin{i},'datatags')  dataTags = varargin{i+1} ;
		else
			fprintf (1, '\nArgument d''entree inconnu\n') ;
			return ;
		end

	end

	% Validation des plages d'operation

   if exist('epochs') & (~isnumeric(epochs) | length(epochs) > 1 | epochs <= 0),
      fprintf (1,'\nLe nombre d''epoques doit etre un entier superieur a 0\n') ;
      return ;
   end

   if ~isnumeric(alpha0) | length(alpha0) > 1 | alpha0 <= 0,
      fprintf (1,'\nLe taux initial d''apprentissage doit etre un nombre superieur a 0\n') ;
      return ;
   end

   if exist('alpha', 'var') & (isempty(alpha) | ~isnumeric(alpha)),
		fprintf (1, '\nAlpha doit etre un vecteur de nombres\n') ;
		return ;
	end

   if exist('alpha', 'var') & ( ~isempty(find(alpha > 1)) | ~isempty(find(alpha < 0)) )
      fprintf (1, '\nAlpha doit etre appartenir a l''intervalle 0 <= alpha <= 1\n') ;
      return ;
   end

   if ~isnumeric(radius0) | length(radius0) > 1 | radius0 < 0,
      fprintf (1,'\nLe rayon initial doit etre un nombre superieur ou egal a 0\n') ;
      return ;
  end
  
   if exist('radius') & (isempty(radius) | ~isnumeric(radius)),
		fprintf (1, '\nLe rayon doit etre un vecteur de nombres\n') ;
		return ;
	end

   if exist('sigma', 'var') & (isempty(sigma) | ~isnumeric(sigma)),
		fprintf (1, '\nSigma doit etre un vecteur de nombres\n') ;
		return ;
	end

   if exist('sigma', 'var') & any( sigma <= 0 ),
		fprintf (1, '\nSigma doit toujours etre superieur a 0\n') ;
		return ;
	end

	if protomapRequested ~= FALSE & protomapRequested ~= TRUE,
		fprintf (1, '\nLe commutateur ''protomap'' doit etre egal a 0 ou 1\n') ;
		return ;
	end

	if datamapRequested ~= FALSE & datamapRequested ~= TRUE,
		fprintf (1, '\nLe commutateur ''datamap'' doit etre egal a 0 ou 1\n') ;
		return ;
	end

	if datamapRequested == TRUE & exist('dataTags'),

      if ~iscellstr(dataTags),
   		fprintf (1, '\nLe nom des individus doit etre une liste de strings\n') ;
	      return ;
      end

      if length(dataTags) ~= size(data,1),
   		fprintf (1, '\nIl n''y a pas autant de noms dans dataTags que d''individus dans data\n') ;
	      return ;
      end
   end
end

% Calcul du nombre d'epoques

if ~exist('epochs')
   epochs = 500 * mapsize(1) * mapsize(2) ;     % par defaut
end

toCompare = epochs ;

if exist('alpha', 'var'),
   toCompare = [ toCompare length(alpha)  ] ;
end

if exist('radius'),
   toCompare = [ toCompare length(radius) ] ;
end

if exist('sigma', 'var'),
   toCompare = [ toCompare length(sigma) ] ;
end

epochs = min (toCompare) ;                      % decision finale


% Valeurs par defaut de alpha, radius et sigma

if ~exist('alpha', 'var'),
   alpha  = linspace ( alpha0, 0, 1+epochs ) ;
end

if ~exist('radius'),
   radius = linspace ( radius0, 0, epochs ) ;
end


if ~exist('sigma', 'var'),

   % Variations possibles :
   %
   % - sigma constant : ~ arbitraire ................... option1
   %                    ~ fonction de mapsize .......... option2
   %                    ~ fonction de radius0 .......... option3
   %
   % - sigma variable : ~ arbitraire ................... option4
   %                    ~ fonction de radius(t) ........ option5

   option1 = TRUE  ;
   option2 = FALSE ;
   option3 = FALSE ;
   option4 = FALSE ;
   option5 = FALSE ;

   if option1,
      sigma = 1*ones(size(radius)) ;
   end

   if option3,
      threshold = .1 ;    % doit etre compris dans ]0,1[

      if radius0 ~= 0,
         sigma0 = sqrt ( (-radius0^2)/log(threshold) ) ;
         sigma  = linspace ( sigma0, 0, 1+epochs ) ;
      else
         sigma = ones(size(radius)) ;
      end
   end

   %dist= 0:.1:max(mapsize) ;
   %plot (dist, exp(-(dist.^2)/sigma^2))

end

% Autres bidules

flagsOfComputedDistanceMaps = zeros (mapsize) ;

if ~exist('dataTags'),
   dataTags = {} ;
end

%==================== Initialisation des poids =========================

[np, vector_length] = size(data) ;     % np pour NumberOfPatterns

if vector_length < 2,
   protomapRequested = FALSE ;         % securite
end

min_data_value = min (min (data)) ;
max_data_value = max (max (data)) ;

rand('state',sum(100*clock)) ;

weights = min_data_value + ...
         (max_data_value - min_data_value) * rand ([mapsize vector_length]) ;

previousWeights = weights ;


% Affichage

if protomapRequested,
   h_figproto = initGraphicProtoMap(data) ;
   drawGraphicProtoMap (h_figproto, weights, 0) ;
end


%===================== Boucle d'apprentissage ==========================

for t = 1:epochs,

   %================ Boucle de lecture des individus ===================

   for patternIndex = 1:np,

      % Activation des neurones de la carte selon le critere 
      % de distance euclidienne (au carre)

      column        = ones  (1,1,vector_length) ;
      column(1,1,:) = data  (patternIndex,:) ;
      data_cube     = repmat(column, mapsize) ;
      activationMap = sum ( (weights - data_cube).^2 , 3) ;

      % Selection du neurone dont les poids sont les plus semblables
      % au vecteur d'entree

      [i,j] = find ( activationMap == min(min(activationMap)) ) ;

      % Calcul du facteur de ponderation radiale

      distanceMap     =  getDistanceMap ([i j], mapsize) ;
      weightsToUpdate = (distanceMap <= radius(t)) ;

      radialAttenuationMap = alpha(t) * exp (-distanceMap/sigma(t)^2) .* weightsToUpdate ;

      for i = 1:vector_length,
         learningMap (:,:,i) = radialAttenuationMap ;
      end

      % Mise a jour des poids au voisinage du neurone gagnant

      weights = weights + learningMap.*(data_cube - weights) ;

   end   % fin de traitement de l'individu

   % Affichage du sous-espace des prototypes

   if protomapRequested,
      drawGraphicProtoMap (h_figproto, weights, t) ;
   end

   % Interruption demandee par l'utilisateur

   if stopEverything
      return ;
   end

   % Securite : eviter le bouclage inutile

   if all(all(all(weights == previousWeights))) & t ~= epochs,
      fprintf (1, '\nEtat stable atteint') ;
      break ;
   else
      previousWeights = weights ;
   end

end   % fin de la boucle d'epoque


%============== Recherche des reponses min et max finales ==============
% ... seulement pour mettre l'echelle de couleur sur le datamap ...
% ... pourrait etre deplace apres la correction des poids de la
%     boucle d'apprentissage ...

if datamapRequested,

   maxResponse = -Inf ;
   minResponse = +Inf ;

   for patternIndex = 1:np,

      % Activation des neurones de la carte selon le critere 
      % de distance euclidienne (au carre)

      column        = ones  (1,1,vector_length) ;
      column(1,1,:) = data  (patternIndex,:) ;
      data_cube     = repmat(column, mapsize) ;
      responseMap   = sum ( (weights - data_cube).^2 , 3) ;

      % Pour affichage ulterieur du datamap seulement

      minAct = min(min(responseMap)) ;
      maxAct = max(max(responseMap)) ;

      minResponse = min ( [minResponse  minAct] ) ;
      maxResponse = max ( [maxResponse  maxAct] ) ;
   end

end


%================= Boucle de classification finale =====================

classes = [] ;    % classe d'appartenance des individus
proto   = [] ;    % prototypes
pMap    = [] ;    % coordonnees des prototypes

classMap   = zeros(mapsize) ;    % assignation des etiquettes de classe
classCount = 0 ;                 % nombre de classes

if datamapRequested,
   h_fig_datamap = initGraphicDataMap (mapsize, np) ;
end

for patternIndex = 1:np,

   % Activation des neurones de la carte selon le critere 
   % de distance euclidienne (au carre)

   column        = ones  (1,1,vector_length) ;
   column(1,1,:) = data  (patternIndex,:) ;
   data_cube     = repmat(column, mapsize) ;
   responseMap   = sum ( (weights - data_cube).^2 , 3) ;

   % Selection du neurone dont les poids sont les plus semblables
   % au vecteur d'entree

   [i,j] = find ( responseMap == min(min(responseMap)) ) ;

   % Memorisation de l'etiquette de classe pour cet individu

   if classMap(i,j) == 0,
      classCount    = classCount + 1 ;
      classMap(i,j) = classCount ;
   end

   % Plus d'un neurone peut satisfaire au critere de distance min ;
   % le cas echeant, reduire classMap(i,j) a sa portion congrue

   if size(i,1) > 1,
      i = i(1) ;
      j = j(1) ;
   end

   classes = [classes ; classMap(i,j)] ;

   % Affichage de la reponse

   if datamapRequested,
      drawGraphicDataMap (h_fig_datamap, patternIndex, responseMap, dataTags) ;
   end

   % Memorisation des reponses

   if nargout == 4,
      rMap{patternIndex} = responseMap ;
   end

end


%================== Boucle de memorisation des prototypes =============

for protoIndex = 1:classCount,

   [i,j] = find ( classMap == protoIndex ) ;
   pMap  = [ pMap  ; i j] ;

   z = [] ;

   for k = 1:vector_length,
      z(k) = weights(i,j,k) ;       % pour lineariser le vecteur
   end

   proto = [ proto ; z ] ;

end


%============================== Fin ====================================

if nargout > 0,
   CLASSES = classes ;
end

fprintf (1, '\nNombre d''epoques : %g\n', t) ;


%----------------------------------------------------------------------
% initGraphicDataMap
% Initialise la figure pour l'affichage de la reponse aux stimuli
%----------------------------------------------------------------------

function h_fig = initGraphicDataMap (mapsize, numberOfPatterns)

   global minResponse maxResponse ;

   tag = ['GraphicDataMap'] ;
   pos = [] ;

   % Detruire une fenetre dataMap, s'il en existe deja une provenant
   % d'une classification anterieure.

   h_fig = findobj ('Tag', tag) ;

   if ~isempty (h_fig),
      pos = get(h_fig, 'Position') ;   % mais preserver la position
      delete (h_fig) ;
   end

   % Creer une nouvelle fenetre, ajuster la palette de couleurs
   % et initialiser les attributs

   h_fig = figure ;

   cmap = jet ;
   cmap(58:64,:) = [] ;            % ... enlever les bleus  fonces
   cmap( 1:7, :) = [] ;            % ... enlever les rouges fonces

   set (h_fig, ...
               'Numbertitle'  , 'off' , ...
               'Menubar'      , 'none', ...
               'DoubleBuffer' , 'on'  , ...
               'Name'         , 'Cartes de reponse des echantillons', ...
               'Tag'          ,  tag  , ...
               'Colormap'     ,  cmap         ) ;

   if ~isempty(pos),
      set (h_fig, 'Position', pos) ;
   end

   % Calcul de la disposition des axes

	nl = floor (sqrt(numberOfPatterns)) ;		   % nombre de lignes
	nc = ceil  (numberOfPatterns/nl) ;			   % nombre de colonnes

   % Map 1D ou 2D ?

   [i,j] = find(mapsize == 1) ;                 % i egale toujours 1

   % Ajustement pour 1D versus 2D

   if ~isempty (j),                             % donc map 1D
      temp       = mapsize(1) ;
      mapsize(1) = mapsize(2) ;
      mapsize(2) = temp ;
   end

   % Initialiser les axes

   for i = 1:numberOfPatterns,

		h_axes = subplot (nl, nc, i) ;
      h_im   = image (zeros(mapsize)) ;

      set (h_axes, ...
               'Tag'            , ['data' int2str(i)] , ...
               'DataAspectRatio', [1 1 1]             , ...
               'FontSize'       ,  6                  , ...
               'XAxisLocation'  , 'top'               , ...
               'XLim'           , [0 mapsize(1)]+.5   , ...
               'YLim'           , [0 mapsize(2)]+.5          )  ;

      % Si map 1D, simplifier les ticks sur la dimension unitaire

      if ~isempty (j),
         if j == 1
            set (h_axes, 'YTick', [1]) ;
         else
            set (h_axes, 'XTick', [1]) ;
         end
      end

   end % for each axis

   % Determiner la position extreme gauche des cellules dans la figure

   h_axes1 = findobj (gcf, 'Tag', 'data1') ;
   pos1    = get (h_axes1, 'Position') ;
   XMin    = pos1(1) ;

   dpos = -0.5 * [ XMin 0 0 0 ] ;    % decalage desire

   % Decaler tout le monde vers la gauche

   for i = 1:numberOfPatterns,

		h_axes = findobj (gcf, 'Tag', ['data' int2str(i)]) ;
      pos    = get (h_axes, 'Position')  ;
      set (h_axes, 'Position', pos+dpos)  ;

   end % for each axis

   % Calculer l'espace disponible a droite

   h_axesc = findobj (gcf, 'Tag', ['data' int2str(nc)]) ;
   posc    = get (h_axesc, 'Position') ;
   XMin    = posc(1) + posc(3) ;

   % Ajouter le colorbar a droite

   h_dummy = axes;                      % axes bidon
   h_axes2 = colorbar ;
   set (h_dummy, 'Visible', 'off') ;

   % Remapper l'echelle en fonction des reponses min et max de la carte

   YTick  = get (h_axes2, 'YTick') ;
   YTick  = [0 YTick] ;
   slope  = (maxResponse-minResponse)/(YTick(length(YTick))-YTick(1)) ;
   YLabel = -YTick*slope + maxResponse ;
   YLabel = round (YLabel*100)/100 ;      % pour enlever l'exces de decimales

   set (h_axes2, ...
               'Tag', 'colorbar', ...
               'Position'  , [XMin+.075 0.3 0.016 0.4] , ...
               'FontSize'  , 6        , ...
               'YTickLabel', YLabel            ) ;
  %               'YTick'     , YTick+1, ...

return

%----------------------------------------------------------------------
% drawGraphicDataMap
%----------------------------------------------------------------------

function drawGraphicDataMap (h_fig, patternIndex, responseMap, dataTags)

   global minResponse maxResponse ;
   global dataVarName ;

   minColorIndex = 1 ;
   maxColorIndex = size (colormap, 1) ;

   % Echelonnage des reponses en fonction de la palette de couleur

   if minResponse ~= maxResponse,

      slope     = (maxColorIndex-minColorIndex) / (minResponse-maxResponse) ;
      scaledMap = round (slope*responseMap + (maxColorIndex-slope*minResponse)) ;

   else
      scaledMap = round (responseMap) ;
   end

   % Trouver le systeme d'axes pour cet echantillon

   h_axes = findobj (h_fig, 'Tag', ['data' int2str(patternIndex)]) ;

   if isempty(h_axes),
      fprintf (1, '\nErreur : aucun systeme d''axes disponible pour le patron %g', patternIndex) ;
      return ;
   end

   % Ecrire les donnees

   if ~isempty(find(size(responseMap) == 1)),
      isMap1D = 1 ;
   else
      isMap1D = 0 ;
   end

   axes  (h_axes) ;
   image ('CData', scaledMap) ;

   if isempty(dataTags),
      title ([ dataVarName '(' int2str(patternIndex) ')']) ;
   else
      title (dataTags{patternIndex}) ;
   end

return

%----------------------------------------------------------------------
% getDistanceMap
% Retourne la carte de distance autour d'un neurone de reference. Pour
% diminuer le temps de calcul, la fonction memorise toutes les cartes
% distinctes dans le vecteur de cellules valuesOfComputedDistanceMaps.
% Au besoin, elle fait calculer la carte par la fonction computeDistanceMap
%----------------------------------------------------------------------

function distanceMap = getDistanceMap (origin, mapsize)

   global flagsOfComputedDistanceMaps ;
   global valuesOfComputedDistanceMaps ;

   i = origin(1) ;
   j = origin(2) ;

   ptr = flagsOfComputedDistanceMaps(i,j) ;

   if ptr ~= 0,
      distanceMap = valuesOfComputedDistanceMaps{ptr} ;
   else

      distanceMap = computeDistanceMap (origin, mapsize) ;
      nextPtr     = 1 + length (valuesOfComputedDistanceMaps) ;

      valuesOfComputedDistanceMaps{nextPtr} = distanceMap ;
      flagsOfComputedDistanceMaps (i,j)     = nextPtr ;

   end

return

%----------------------------------------------------------------------
% computeDistanceMap
% Calcule la distance euclidienne sur une carte 2D
% autour d'un neurone de reference
%----------------------------------------------------------------------

function distanceMap = computeDistanceMap (origin, mapsize)

   for i = 1:mapsize(2),
      xdist(i) = i - origin(2) ;
   end

   for j = 1:mapsize(1),
      ydist(j) = j - origin(1) ;
   end

   [xdist, ydist] = meshgrid (xdist, ydist) ;
   distanceMap = sqrt ( xdist.^2 + ydist.^2 ) ;

return


%----------------------------------------------------------------------
% initGraphicProtoMap
% Initialise l'affichage des prototypes sur la carte de Kohonen.
%----------------------------------------------------------------------

function h_fig = initGraphicProtoMap (data)

   % Verification des extremes

   minValues = min (data) ;
   maxValues = max (data) ;

   minX = minValues(1) ;
   minY = minValues(2) ;
   maxX = maxValues(1) ;
   maxY = maxValues(2) ;

   % Initialisation de la figure

   tag = ['GraphicProtoMap'] ;
	back_color = [1 1 1] ;
   h_fig = findobj ('Tag', tag) ;

   if isempty (h_fig),
      h_fig = figure ;     % definir une nouvelle figure
   end

	set (h_fig, 'Numbertitle' , 'off'       , ...
					'Menubar'     , 'none'      , ...
               'Tag'         ,  tag        , ...
					'Color'       ,  back_color , ...
					'DoubleBuffer', 'on'                 ) ;

   set (gca,   'XLim'        , [minX maxX] , ...
               'YLim'        , [minY maxY]          ) ;

return

%----------------------------------------------------------------------
% drawGraphicProtoMap
% Affichage de la carte de Kohonen des prototypes.  Seules les deux 
% premieres composantes sont utilisees
%----------------------------------------------------------------------

function drawGraphicProtoMap (h_fig, weights, epoch)

   global protomapRequested stopEverything ;

   % Fin demandee par l'usager ?

   objects = findobj ;

   if isempty( find(objects == h_fig))

      ans = menu ('Que voulez-vous faire ?', ...
                  'Arreter l''affichage des prototypes, mais continuer la simulation',...
                  'Terminer net fret sec') ;

      if ans == 1,
         protomapRequested = 0 ;
      else
         stopEverything = 1 ;
      end

      return ;
   end

   % ... non

   figure (h_fig) ;        % activer la figure

   axesLimits = axis ;     % preserver les axes
   cla ;                   % enlever les objets

   % Valeurs a afficher

   xValues = weights(:,:,1) ;       % forme matricielle
   yValues = weights(:,:,2) ;

   [i,j] = size(xValues);

   for k = 1:i*j,                   % forme vectorielle
      x(k) = xValues(k) ;
      y(k) = yValues(k) ;
   end

   % Verification : carte 1D ou 2D ?

   if any( size(xValues) == 1)
      map1D = 1 ;
   else
      map1D = 0 ;
   end

   % Mise a jour de la carte

   if map1D,
      plot (x, y, 'r-', x, y, 'bo') ;

   else
      zValues = zeros (size(xValues)) ;
      surface (xValues, yValues, zValues,...
               'FaceColor', 'none', ...
               'EdgeColor', [1 0 0]  ) ;

      hold on
      scatter(x, y) ;
      hold off
   end

   set    (h_fig, 'Name', ['Sous-espace des prototypes (epoque ' int2str(epoch) ')']) ;

   axis   (axesLimits) ;

   h_xstr = xlabel ('x1') ;
   h_ystr = ylabel ('x2') ;

   FontSize = 7 ;
   set    (gca   , 'FontSize', FontSize) ;
   set    (h_xstr, 'FontSize', FontSize) ;
   set    (h_ystr, 'FontSize', FontSize) ;

   drawnow ;

return
