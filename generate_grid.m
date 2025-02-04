function grid = generate_grid(domain,Npts)
% generate a grid in nD. Npnts in each dimension
%split domain matrix to cell array
domain_cell = mat2cell(domain,2,ones(1,size(domain,2)));
%call linspace for each cell element (i.e. each dimensions)
grid = cellfun(@(x) linspace(x(1),x(2),Npts), domain_cell ,...
  'UniformOutput',false);
%call ndgrid and capture output
[grid{:}] = ndgrid(grid{:});
%grid is now a cell of tensors with Npts^size(domain,2) elements
%columnstack all vectrs
grid = cellfun(@(x) x(:), grid, 'UniformOutput', false);
%and collect into a since column matrix with one point per line
grid = cell2mat(grid);
end