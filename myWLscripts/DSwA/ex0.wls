(* Exercices for Chapter 0 *)
Print[" --- "]
Print["Exercice 1: Arithmetic"]
Print[" --- "]

Print[4+5-6]
Print[3^12]
Print[Sin[0.1*Pi]]
Print[(2-(3-4(3+7(1-(2(3-5))))))]
Print[(2/5)-(3/4)*(2/3)]

Print[" --- "]
Print["Exercice 2: Matrices"]
Print[" --- "]

mA = { {1,2,-1}, {0,1,0}, {3,-1,2}};
mB = { {1,2,3}, {1,1,2}, {0,1,2}};
mC = { {2,1,1}, {0,1,-1}, {4,2,2}};
Print[mA + 4*(mB.mC)]

Print["Inverse of A = ", Inverse[mA]]
Print["Inverse of B = ", Inverse[mB]]
Print["Inverse of C = ", Inverse[mC]]

Print[mA*mA*mA]

Print[" det C = ", Det[mC]]

Print["Eigs B"]
Print[N[Eigenvalues[mB]]]
Print[N[Eigenvectors[mB]]]

Print[" --- "]
Print["Exercice 3: Complex numbers"]
Print[" --- "]

z1 = 1+1*I;
z2 = -2 + 1*I;
z3 = -1*I;

Print[z1 + z2 - z3]
Print[(z1*z2)/z3]
Print[ComplexExpand[Exp[z1]]]
Print[Log[z1]]
Print[ComplexExpand[Sin[z3]]]

Print[" --- "]
Print["Exercice 4: Limits"]
Print[" --- "]

Print[Limit[Sin[x]/x,x->0]]
Print[Limit[(x^3+3*x^2-5)/(2*x^3-7*x),x->Infinity]]
Print[Limit[(Cos[x]+1)/(x-Pi),x->0]]
Print[Limit[1/x,x->0,Direction->"FromAbove"]]
Print[Limit[(2*Sinh[x]-2*Sin[x])/(Cosh[x]-1),x->0]]

Print[" --- "]
Print["Exercice 5: Derivatives"]
Print[" --- "]

Print[D[3*x^3 + 2*x^2 - 5,x]]
Print[D[Sqrt[1+x^4],x]]
Print[D[E^x * Sin[x] * Cos[x],x]]
Print[D[Tanh[x],x]]
Print[D[x^(Log[x]),x]]

