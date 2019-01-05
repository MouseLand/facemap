% normalize columns of matrix
function x = normc(x)

x = bsxfun(@times, x, 1./(sum(x.^2,1).^0.5));