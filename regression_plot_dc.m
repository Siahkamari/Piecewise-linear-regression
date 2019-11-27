function regression_plot_dc(X,y,f_reg, f_hat)
[n, dim] = size(X);

%% Plotting
d_mesh = 0.01;
figure
col = get(gca,'colororder');

if dim==1
    X_mesh = (0:d_mesh:1)';
    
    subplot(1,2,1)
    plot(X_mesh, f_hat(X_mesh)); hold on
    scatter(X,y,100,col(2,:),'filled'); hold off
    
    subplot(1,2,2)
    plot(X_mesh, f_reg(X_mesh)); hold on
    scatter(X,y,100,col(2,:),'filled'); hold off
    
elseif dim > 1
    if dim > 2
        warning("dimmension > 2! we plot just 2 features")
    end
    [X1mesh,X2mesh] = meshgrid(0:d_mesh:1,0:d_mesh:1);
    X_mesh = [X1mesh(:), X2mesh(:)];
    
    f_hat_mesh = reshape(f_hat([X_mesh,0*repmat(mean(X(:,3:end)), size(X_mesh,1),1)]) , size(X1mesh));
    f_reg_mesh = reshape(f_reg([X_mesh,0*repmat(mean(X(:,3:end)), size(X_mesh,1),1)]) , size(X1mesh));
    
    subplot(1,2,1)
    meshc(X1mesh,X2mesh, f_hat_mesh); hold on
    scatter3(X(:,1),X(:,2),y,100,col(2,:),'filled'); hold off
    
    subplot(1,2,2)
    meshc(X1mesh,X2mesh, f_reg_mesh); hold on
    scatter3(X(:,1),X(:,2),y,100,col(2,:),'filled'); hold off
end
drawnow;

