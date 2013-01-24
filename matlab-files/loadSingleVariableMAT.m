function y = loadSingleVariableMAT(filename)

foo = load(filename);
whichVariables = fieldnames(foo);
if numel(whichVariables) == 1
    y = foo.(whichVariables{1});
else
    error('MAT file %s contains more than one variable\n',filename);
end

end

