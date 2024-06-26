img = imread("real_results\rgb_results\dwmt\frame1channel23.png");
%% The position and border width of the area that needs to be enlarged
[patch_x, patch_y, patch_width] = deal(451, 388, 100);
[M,N,D] = size(img);
x = 141; % The position of the enlarged area placement
y = 321;
magnification = 3; % Magnification

line_width = 2; % Enlarged area border width
color = [0, 255, 0];  % The color of the border
img = DrawRectangle(patch_x, patch_y, patch_width, x, y, line_width, color, magnification, img);
fig = imshow(img);

function img = DrawRectangle(patch_x, patch_y, patch_width, x, y, line_width, color, magnification, img)
    X1 = patch_x;
    Y1 = patch_y;
    X2 = X1+patch_width-1;
    Y2 = Y1+patch_width-1;
    img_patch = img(Y1:Y2,X1:X2 , :);
    img(Y1:Y2,   X1:X1+line_width, 2) = color(2);
    img(Y1:Y2,   X1:X1+line_width, 1) = color(1);
    img(Y1:Y1+line_width, X1:X2, 2) = color(2);
    img(Y1:Y1+line_width, X1:X2, 1) = color(1);
    img(Y1:Y2,   X2-line_width-1:X2, 2) = color(2);
    img(Y1:Y2,   X2-line_width-1:X2, 1) = color(1);
    img(Y2-line_width:Y2, X1:X2, 2) = color(2);
    img(Y2-line_width:Y2, X1:X2, 1) = color(1);
    img(Y1:Y2,   X1:X1+line_width, 3) = color(3);
    img(Y1:Y2,   X1:X1+line_width, 3) = color(3);
    img(Y1:Y1+line_width, X1:X2, 3) = color(3);
    img(Y1:Y1+line_width, X1:X2, 3) = color(3);
    img(Y1:Y2,   X2-line_width:X2, 3) = color(3);
    img(Y1:Y2,   X2-line_width:X2, 3) = color(3);
    img(Y2-line_width:Y2, X1:X2, 3) = color(3);
    img(Y2-line_width:Y2, X1:X2, 3) = color(3);
    [M,N,D] = size(img);
    [M1,N1,D1] = size(img_patch);
    img_patch(1:M1, 1:line_width, 1)=color(1);
    img_patch(1:M1, 1:line_width, 2)=color(2);
    img_patch(1:M1, 1:line_width, 3)=color(3);
    img_patch(1:M1, N1-line_width+1:N1, 1)=color(1);
    img_patch(1:M1, N1-line_width+1:N1, 2)=color(2);
    img_patch(1:M1, N1-line_width+1:N1, 3)=color(3);
    img_patch(M1-line_width+1:M1, 1:N1, 1)=color(1);
    img_patch(M1-line_width+1:M1, 1:N1, 2)=color(2);
    img_patch(M1-line_width+1:M1, 1:N1, 3)=color(3);
    img_patch(1:1+line_width-1, 1:N1, 1)=color(1);
    img_patch(1:1+line_width-1, 1:N1, 2)=color(2);
    img_patch(1:1+line_width-1, 1:N1, 3)=color(3);
    J = imresize(img_patch,magnification);
    [M2,N2,D2] = size(J);
    img(y:y+N2-1, x:x+M2-1,:) = J;
end