# Notes (on computer theory)

## Basic Rules

1. Be careful while compiling. `minted` package requieres to modify your LaTeX compiling recipe.
2. Do not modify `.gitignore` unless necessary.
3. ...

## Writing  rules

1. Use `\tt` font for machine alphabet. E.g. `\Sigma = \{ {\tt a}, {\tt b}, {\tt c} \}`.
2. When labeling, use the scheme `Topic#TexSection-Enviroment-BriefDescription`. E.g. `\label{CT1-D-DFA}`.
    * `D` definition,
    * `C` corollary,
    * `P` proposition,
    * `L` lemma,
    * `T` theorem,
    * `E` example,
    * ...
    * `S` section, 
    * `SS` subsection.

## Images rules

1. Make a `standalone` file in `\fotos\`, then compile. Name it `Topic#TexSection-#.tex`. E.g. `CT1-1.tex`.
2. Label any picture as its file name. E.g. `\label{CT1-1}`.
3. Use `\begin{figure}[hb]`.
4. ...
