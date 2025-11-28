function suite = geometric (init, taux, t)

% GEOMETRIC
%                  suite = geometric (init, taux, t)
%
% Creation d'une suite geometrique commencant a une valeur initiale init,
% et dont chaque element est egal a une fraction de l'element precedent :
%
%                    suite(t+1) = suite(t) * taux
% Exemple : 
%
%     geometric(.8, .5, 4) = [.8*.5^0  .8*.5^1  .8 *.5^2  .8*.5^3]
%                          = [.8       .4       .2        .1     ]
%

%============================ Securite ===============================

if nargin < 3,
   fprintf (1, '\nNombre insuffisant d''arguments en entree\n') ;
   return ;
end

%======================== Calcul de la suite =========================

suite(1) = 1 ;

for i = 2:t
   suite (i) = suite(i-1) * taux ;
end

suite = init * suite ;
