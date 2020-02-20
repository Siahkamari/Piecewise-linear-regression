
% Makes a gif of plots from iterations of forward phase. Plus the final
% frame is made after backward pruning phase and after the model is turned
% into piecewise-cubic.

% The gif is saved as 'iterations.gif' in the default/current Matlab directory.

% This is a simple, not very efficient implementation. It simulates
% building of one model by building lots of models of increasing size.
% Crazy, I know.

clear
[X1,X2] = meshgrid(-1:0.2:1, -1:0.2:1);
X(:,1) = reshape(X1, numel(X1), 1);
X(:,2) = reshape(X2, numel(X2), 1);
clear X1 X2;
Y = sin(0.83*pi*X(:,1)) .* cos(1.25*pi*X(:,2));

noisy = false; % set to true to add noise
if noisy
    rng(1);
    Y = Y + randn(121,1) .* 0.2;
end

%%

% To choose these maxFuncs values, I first experimentally found out at what
% model size the forward phase either terminates or gets to GCV=Inf.
% In any case, further code expects maxFuncs to be odd number.
if noisy
    maxFuncs = 47;
    c = 3;
else
    maxFuncs = 89;
    c = 0;
end

for i = 1 : 2 : (maxFuncs+2) % step size is 2 because the basis functions are added in pairs
    if i <= maxFuncs
        % Forward phase works with piecewise-linear models
        params = aresparams2('maxFuncs', i, 'c', c, 'maxInteractions', 2, 'cubic', false, 'prune', false);
    else
        % The last frame has pruned model which is turned into piecewise-cubic
        params = aresparams2('maxFuncs', maxFuncs, 'c', c, 'maxInteractions', 2, 'cubic', true, 'prune', true);
    end
    model = aresbuild(X, Y, params);
    
    % Plotting model
    fh = aresplot(model, [1 2], [], [], [], [], false, 'ZLim', [-1.5 1.5]);
    set(fh, 'Color', 'w');
    
    % Plotting data
    hold on;
    plot3(X(:,1), X(:,2), Y, 'or', 'MarkerSize', 5, 'LineWidth', 1);
    
    set(gca, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', [], 'ZLim', [-1.5 1.5]);
    
    % Adding annotation
    if i <= maxFuncs
        str = {['Forward phase iteration ' num2str((i-1)/2)], ...
              [num2str(length(model.coefs)) ' basis functions. Piecewise-linear.']};
    else
        str = {'Backward pruning phase', ...
              [num2str(length(model.coefs)) ' basis functions. PIECEWISE-CUBIC.']};
    end
    annotation('textbox', [0.13 0.92 0 0], 'String', str, 'FitBoxToText', 'on', 'BackgroundColor', 'w');
    
    % Saving to file
    ax = gca;
    ax.Units = 'pixels';
    pos = ax.Position;
    rect = [-2, -2, pos(3)+4, pos(4)+3];
    drawnow
    im = frame2im(getframe(ax, rect));
    %im = frame2im(getframe(1)); % We could also save the figure without cropping
    [imind, cm] = rgb2ind(im, 64);
    filename = 'iterations.gif'; % Should be saved to the default/current Matlab directory
    if i == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.25);
    elseif i <= maxFuncs
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.25);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 2); % Last frame has longer delay
    end
    
    close(fh);
end
