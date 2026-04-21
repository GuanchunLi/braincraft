1. CHECKED -- naming of variables of direction circuit: current cos_n and sin_n is confusing; can we correct that? current `cos_n` should be `sin_n` (also flipping the sign if possible); and current `sin_n` should be `cos+_n`. Change all related formulas. 
2. Let's still focus on improving reflex bio 2 player. For the reflex features (slot 0,1, 2, 3, 4), I have following questions:
(i) hit_feat seems not being used across the run? If so, please remove it and update all the indexes correspondingly. A
(ii) Do we need 100*k_sharp or k_sharp itself is enough? 
(iii) Is possible to merge prox_left and safe_left (same for prox_right and safe_right)?
3. CHECKED -- Explain more on reward circuit? (what's the role of armed_katch)