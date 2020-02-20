function regression_plot(X, y, f1, f2, f3)
[n, dim] = size(X);

%% Plotting
d_mesh = 0.01;
col = get(gca,'colororder');

if dim==1
    X_mesh = (min(X):d_mesh:max(X))';
    
    plot(X_mesh, f1(X_mesh)); hold on
    plot(X_mesh, f2(X_mesh));
    plot(X_mesh, f3(X_mesh));
    scatter(X,y,5,'filled');
    ylabel("$y$","Interpreter", "latex","FontSize", 20)
    xlabel("$x$","Interpreter", "latex","FontSize", 20)
    legend("$f(x)$","$\hat f_{\lambda_0}(x)$","$\hat f_{\lambda_{CV}}(x)$","Interpreter", "latex","FontSize", 20)
    
elseif dim > 1
    if dim > 2
        warning("dimmension > 2! we plot just 2 features")
    end
    [X1mesh,X2mesh] = meshgrid(min(X(:,1)):d_mesh:max(X(:,1)),min(X(:,2)):d_mesh:max(X(:,2)));
    X_mesh = [X1mesh(:), X2mesh(:)];
    
    f1_mesh = reshape(f1([X_mesh,0*repmat(mean(X(:,3:end)), size(X_mesh,1),1)]) , size(X1mesh));
    f2_mesh = reshape(f2([X_mesh,0*repmat(mean(X(:,3:end)), size(X_mesh,1),1)]) , size(X1mesh));
    f3_mesh = reshape(f3([X_mesh,0*repmat(mean(X(:,3:end)), size(X_mesh,1),1)]) , size(X1mesh));
    
    figure(1)
    meshc(X1mesh,X2mesh, f1_mesh); hold on
    scatter3(X(:,1),X(:,2),y,100,col(2,:),'filled')
    zlabel("$f(x)$","Interpreter", "latex","FontSize", 20)
    xlabel("$x_1$","Interpreter", "latex","FontSize", 20)
    ylabel("$x_2$","Interpreter", "latex","FontSize", 20)
    xticks([]), yticks([])
    
    figure(2)
    meshc(X1mesh,X2mesh, f2_mesh); hold on
    scatter3(X(:,1),X(:,2),y,100,col(2,:),'filled'); 
    zlabel("$\hat f_{\lambda_0}(x)$","Interpreter", "latex","FontSize", 20)
    xlabel("$x_1$","Interpreter", "latex","FontSize", 20)
    ylabel("$x_2$","Interpreter", "latex","FontSize", 20)
    xticks([]), yticks([])
    
    figure(3)
    meshc(X1mesh,X2mesh, f3_mesh); hold on
    scatter3(X(:,1),X(:,2),y,100,col(2,:),'filled')
    zlabel("$\hat f_{\lambda_{CV}}(x)$","Interpreter", "latex","FontSize", 20)
    xlabel("$x_1$","Interpreter", "latex","FontSize", 20)
    ylabel("$x_2$","Interpreter", "latex","FontSize", 20)
    xticks([]), yticks([])
end

