function [xkap]=multifLTANNoutputs3(pgbest,x,p,s)
n=length(x);
w=pgbest(1:p);
b=pgbest(p+1:2*p);
alfa1=pgbest(2*p+1);
beta1=pgbest(2*p+2);
alfa2=pgbest(2*p+3);
alfa3=pgbest(2*p+4);
o2(1)=0;
for j=1:p+s
    xkap(j)=x(j);
    y(j)=0;
end
for i=p+s+1:n+1
    o1(i)=alfa3*(alfa1*x(i-1)+(1-alfa1)*xkap(i-1))+(1-alfa3)*(alfa2*x(i-s)+(1-alfa2)*xkap(i-s));
    o3(i)=w(1)*x(i-1)+b(1);
    if p>1
        for j=2:p
            o3(i)=o3(i)*(w(j)*x(i-j)+b(j));
        end
    end
    o3(i)=(1/(1+exp(-o3(i))));
    xkap(i)=beta1*o1(i)+(1-beta1)*o3(i);
end
xkap=xkap(1:end-1);
end