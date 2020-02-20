X = 4*rand(10,1)-2;

y = abs(X).^2;


[f_hat, params] = c_fit(y, X, 0.0001);

sum(abs(y-f_hat(X)))

x = [-2:0.01:2]';

area(x,4 - f_hat(x), 'FaceColor',[0.7,0.7,0.7])

set(gca,'Ydir','reverse'); hold on

Y = params.phi' + x*params.grad';

for i = 1:length(params.phi)
    
    plot(x, 4- Y(:,i),'LineWidth',2);
end

ylim([0,4])
xlim([-2,2])
xticks([])
yticks([])
xlabel('x',"FontSize",20)
ylabel('y',"FontSize",20)
text(0,2,'y = x^2', "FontSize",35)
