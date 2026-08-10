
x = linspace(0.001, 20, 10000);
arr = compare_w_and_transformed_w(x);


function return_arr = compare_w_and_transformed_w(x)
    % Inputs:
    %   x = array on which to plot, x-axis of w-values

    w_normal = normcdf(x, "upper");
    figure;
    hold on;
    plot(x, w_normal, 'DisplayName', 'Normal $w$-test, q=1');

    return_arr = cell(5, 2); % Initialize the return array
    wj_arr = linspace(0,4,5);
    % Tq = w_i^2 + \bar{w}_j^2 for q=2
    for i = 1:5
        wj = wj_arr(i);
        Tq = x.^2 + wj^2;
        cdf_val = chi2cdf(Tq, 2, "upper"); % actually 1-cdf, become much more accurate
        w_i_trans = norminv(1.0-cdf_val);
        

        plot(x, cdf_val, 'DisplayName', ['$w_i(T_q)$ with $\overline{w}_j=' num2str(wj) '$']);
        return_arr{i+1, 1} = cdf_val;
        return_arr{i+1, 2} = w_i_trans;
    end

    xlim([0, 20]);
    ylim([-5, 20]);
    xlabel('$w_i$ value', 'Interpreter', 'latex');
    ylabel('1-cdf for transformed $T_q$ or $w$ value', 'Interpreter', 'latex');
    legend('Interpreter', 'latex');
    set(gca, 'YScale', 'log')
    hold off;
end