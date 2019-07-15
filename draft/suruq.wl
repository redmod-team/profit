(* ::Package:: *)

(* ::Section:: *)
(*Nonlinear transform of a Gaussian random variable*)


(* ::Input:: *)
(*f[u_]=(4u)^2+1;*)
(*pu=Exp[-u^2];*)
(*pycondu=1/Sqrt[2Pi eps^2] Exp[-((y-f[u])^2/(2eps^2))]/.eps->0.1;*)
(*py=NIntegrate[pycondu*pu,{u,-Infinity,Infinity}];*)
(*Plot[pu,{u,-3,3},AxesLabel->{"u","p[u]"}]*)
(*Plot[py,{y,-3,3},PlotRange->{0,1},AxesLabel->{"y","p[y]"}]*)



