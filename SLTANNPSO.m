function [xkapong,pgbest,yhattestg,rmsetest]=SLTANNPSO(xt,pm,s,nf,ntest)
% Training of the network
itr=1000;ps=30;vmaps=1;
w1=0.4;w2=0.9;c1i=2;c2i=1;c1f=1;c2f=2;
msebestEski=10^6;kk=0;
xorj=xt;
xt=xorj(1:end-ntest);
x=(xt-min(xt))/(max(xt)-min(xt));
saat=clock;
rand('seed',saat(6)*10000000);
rn=rand('seed');
p=2*pm+4;
%p parametre sayýsý
n=length(x);
A=unifrnd(0,1,ps,p);
V=unifrnd(-vmaps,vmaps,ps,p);
%Pso paramterleri program hýzlý çalýþmasý için baþta oluþturuluyor;
w=zeros(1,itr);
c1=zeros(1,itr);
c2=zeros(1,itr);
for k=1:ps
    yhat=multifLTANNoutputs3(A(k,:),x,pm,s);
    nh=length(yhat);
    for i=1:nh
        hata(i)=(i/nh)*(yhat(i)-x(n-nh+i))^2;
    end
    mse(k)=mean(hata);
end
%en iyi parçacýðý saklýyor.
MSEegt=min(mse);
for i=1:ps
    if MSEegt==mse(i)
        dd=i;
        break
    end
end
pgbest=A(dd,:);
msebest=mse(dd);
pid=A;
msepid=mse;
i22=0;
for i1=1:itr
    i22=i22+1;
    % Güncelleme
    w(i1)=(w1-w2)*((itr-i1)/itr)+w2;
    c1(i1)=(c1f-c1i)*(i1/itr)+c1i;
    c2(i1)=(c2f-c2i)*(i1/itr)+c2i;
    for i2=1:ps
        for i3=1:p
            V(i2,i3)=V(i2,i3)*w(i1)+c1(i1)*unifrnd(0,1)*(pid(i2,i3)-A(i2,i3))+c2(i1)*unifrnd(0,1)*(pgbest(i3)-A(i2,i3));
            V(i2,i3)=min(vmaps,max(-vmaps,V(i2,i3)));
            A(i2,i3)=A(i2,i3)+V(i2,i3);
        end
        A(i2,p-3)=min(1,max(0,A(i2,p-3)));
        A(i2,p-2)=min(1,max(0,A(i2,p-2)));
        A(i2,p-1)=min(1,max(0,A(i2,p-1)));
        A(i2,p)=min(1,max(0,A(i2,p)));
    end
    if i22>=30
        A=unifrnd(0,1,ps,p);
        V=unifrnd(-vmaps,vmaps,ps,p);
        i22=0;
    end
    for k=1:ps
        yhat=multifLTANNoutputs3(A(k,:),x,pm,s);
        nh=length(yhat);
        for i=1:nh
            hata(i)=(i/nh)*(yhat(i)-x(n-nh+i))^2;
        end
        mse(k)=mean(hata);
    end
    %en iyi parçacýðý saklýyor.
    MSEegt=min(mse);
    for i=1:ps
        if MSEegt==mse(i)
            dd=i;
            break
        end
    end
    if MSEegt<msebest
         pgbest=A(dd,:);
         msebest=mse(dd);
         %[i1,msebest]
    end
    for j1=1:ps
        if mse(j1)<=msepid(j1)
            pid(j1,:)=A(j1,:);
            msepid(j1)=mse(j1);
        end
    end 
    if abs((msebestEski-msebest)/msebest)<10^-3
        kk=kk+1;       
    else
        kk=0;
    end
    if kk>20
            break
    end
    msebestEski=msebest;
end

itn=i1;
%Test aþamasý
x=(xorj-min(xt))/(max(xt)-min(xt));
yhattum=multifLTANNoutputs3(pgbest,x,pm,s);
nt=length(x);
yhattest=yhattum((nt-ntest+1):nt);
yhattestg=(yhattest)*(max(xt)-min(xt))+min(xt);
xtest=xorj(end-ntest+1:end);
for i=1:ntest
    hata3(i)=(xtest(i)-yhattestg(i))^2;
end
rmsetest=(mean(hata3))^0.5;    
%Öngörü aþamasý
xkapong=zeros(nf,1);
for j=1:nf
    xkap=multifLTANNoutputxs3(pgbest,x,pm,s);
    xkapong(j)=xkap*(max(xt)-min(xt))+min(xt);
    x=[x;xkap];
end
end