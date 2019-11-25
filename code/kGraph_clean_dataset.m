%Ignacio Garrido Botella
%Code for reading and cleaning the datasets of UmichIndoorCorridor2012
%Files saved in .mat format

%%
%Variables
rows = 400;
cols = 695;
dir_new_files = '/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/datasets/UmichIndoorCorridor2012/clean_datasets';
%%
%Save Files
names_files = ["Dataset_+" "Dataset_L" "Dataset_T_1" "Dataset_T_2"];
count = 0;
for file = names_files
    dir_gt = strcat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/datasets/UmichIndoorCorridor2012/ground_truth/', file, '/');    
    dir_pic = strcat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/datasets/UmichIndoorCorridor2012/dataset/', file, '/');
    matfiles = dir(fullfile(strcat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/datasets/UmichIndoorCorridor2012/ground_truth/', file, '/*.mat')));
    data_gt = zeros(length(matfiles),rows, cols);
    data_pic = uint8(zeros(length(matfiles),rows, cols,3));
    for x = 1:length(matfiles)  
      fid = load(strcat(dir_gt, matfiles(x).name));
      image = imread(strcat(dir_pic, strcat(matfiles(x).name(1:end-4),'.ppm')));
      matr = fid.ground_truth;
      for i =  1:rows
          for j = 1:cols 
             if matr(i,j) ~= -1
                 matr(i,j) = 0;
             else
                 matr(i,j) = 1;
             end
          end
      end
      data_gt(x,:,:) = matr;
      data_pic(x,:,:,:) = image;
    end
    count = count + 1
    save(strcat(dir_new_files, '/dataset_',string(count),'_gt.mat'),'data_gt');
    save(strcat(dir_new_files, '/dataset_',string(count),'_im.mat'),'data_pic');
end


  
