files = ls("/mnt/Data/summer2013/gestures/training1/*.mat");

output = fopen("output.csv", "w");

for i = 1:length(files);,
    
    filename = [];
    filename = files(i, :);
    contents = load(filename);
    
    for l = 1:length(contents.Video.Labels);,
        fprintf(output, "(%s, %s, %d, %d)\n", filename, contents.Video.Labels(l).Name, contents.Video.Labels(l).Begin, contents.Video.Labels(l).End);
    
    end
end



