function SphericalVisualisationIDS(A,Qy)
%% Description
% This function visualizes the misclosure space partitioning, for datasnooping when r=3, in
% 3D space. The partitioning regions are projected on the surface of a unit
% sphere. 

%Author: Safoora Zaminpardaz

%% Inputs
% A   : m x n design matrix
% Qy : m x m observation VCV matrix 


%% Computing B-matrix and misclosure VCV matrix
m = size(A,1);
k=10; % nr of hypotheses in IDS = 2
B = null(A');
Qt = B'*Qy*B;
%Transforming t to a misclsure with a VCV matrix equal to the identity matrix
[V,D] = eig(Qt); 
B = B*V*(diag(diag(D^-0.5)))*V';

%% Building a grid of points on the surface of a zero-centred unit sphere (3D)
t1 = 0:0.01:2*pi;
t2 = -pi/2:0.01:pi/2;
[tet,si] = meshgrid(t1,t2);
size(tet)
size(si)

pathname = 'H:\My Documents\PhD\Python\Case Study\IDS\Sim Data\';
x = importdata(append(pathname, 'grid_x.txt'));
y = importdata(append(pathname, 'grid_y.txt'));
z = importdata(append(pathname, 'grid_z.txt'));
factor = 6;

filename = append(pathname, 't_3D_data_factor_', int2str(factor), '.txt');
fprintf(filename)
t_3D = importdata( filename);

identifications = importdata(append(pathname, 'corresponding_identifications_factor_', int2str(factor) ));

v = [x(:)';y(:)';z(:)'] %collection of unit vectors in 3D
mv = size(v,2);

%% Computing w-tests for all the points in the generated grid

% ct = B';
% w = zeros(m,mv);
% for i = 1 : m
%     cti = ct(:,i);
%     cinorm = vecnorm(cti);
%     w(i,:) = abs((cti'*v)/cinorm);
% end
% [~,u] = max(w,[],1);%returning the index of the selected hypotheses based upon samples of t

u = identifications;
%% Plotting the partitioning regions projected on the unit sphere
figure
surf(x,y,z,'EdgeColor','none','FaceColor','k');
hold on
for i = 1 : k
    xx = x(:);
    xx(u~=i) = nan;
    xx = reshape(xx,size(x,1),size(x,2));
    
    yy = y(:);
    yy(u~=i) = nan;
    yy = reshape(yy,size(x,1),size(x,2));
    
    zz = z(:);
    zz(u~=i) = nan;
    zz = reshape(zz,size(x,1),size(x,2));    
    
    h(i)=surf(xx,yy,zz,'EdgeColor','none','FaceColor',[1 1 1]*i/15);
    p_name{i} = ['$\mathcal{P}_{' int2str(i) '}$'];
end

%% Plotting the ending points of the normalized cti vectors
% for i = 1:m
%     cti = ct(:,i)/vecnorm(ct(:,i));
%     plot3(cti(1),cti(2),cti(3),'Color','k','Marker','o','LineWidth',7,'MarkerSize',6)
%     plot3(-cti(1),-cti(2),-cti(3),'Color','k','Marker','o','LineWidth',7,'MarkerSize',6)
% end

legend(h,p_name,'Interpreter','latex','box','off','FontSize',14)
axis equal
% axis off

%% Alternative way of plotting
% figure
% hold on
% for i = 1 : m
%     h(i)= plot3(v(1,u==i),v(2,u==i),v(3,u==i),'Color',[1 1 1]*i/7,'Marker','.','LineStyle','none');
% end
% legend(h,p_name,'Interpreter','latex','box','off','FontSize',14)
% axis equal
% axis off








