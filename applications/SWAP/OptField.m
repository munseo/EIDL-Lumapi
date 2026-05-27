
close all;
clear
clc
center_wl=1.55
d_rs=50
d_height=0.4
d_length=8.0
d_width=8.0

pt_2_cm=37.8;
XY_w=16.5;
XY_h=2.0;


% cd 0_Near_fwd
% cd 1_Near_flip
% cd 2_15um_mod
% cd 3_2um_sym
% cd 4_10_20um
% cd 5_25um
% cd Noise_added\A\

noise=0;

                    X0fig=0; Y0fig=0;
                    Wfig=1500;Hfig=700; 

    set(groot,'defaultAxesFontName','Arial')
    set(groot,'defaultAxesFontSize',5.3)
    % set(groot,'defaultAxesTitleFontSizeMultiplier',1);
    set(groot,'defaultAxesLabelFontSizeMultiplier',1.2);
    set(groot,'DefaultLineLineWidth',0.8)
    set(groot,'DefaultAxesLineWidth',0.8)
    set(groot,'defaultTextFontSize',5.3)  

figure('Name','Results','position',[500 100 pt_2_cm*4.5 pt_2_cm*8.3],'color','w')
for output_load=0:1
            real_tran=load("Real_flux_"+output_load+"_to_"+abs(output_load-1)+".txt");
            real_purity=load("Real_purity_"+output_load+"_to_"+abs(output_load-1)+".txt");
            if noise==1
                real_noise=load("Real_noise_"+output_load+"_to_"+abs(output_load-1)+".txt");
            end
            real_wl=1./load("Real_freqs_"+output_load+"_to_"+abs(output_load-1)+".txt");
            wl_var=abs(real_wl-center_wl);
            center_idx=find(wl_var==min(wl_var));
                if output_load==1
                    p1=plot(real_wl, (real_purity),'o-r','LineWidth', 1);
                    hold on
                    p2=plot(real_wl, (real_tran),'o-b','LineWidth', 1);
                    p3=plot(real_wl, (real_tran.*real_purity),'--k','LineWidth', 1);
                    if noise==1
                        plot(real_wl, (real_noise),'-m','LineWidth', 1)
                    end
                % lgd=legend([p1,p2,p3],'Mode purity','Transmittance','Conversion efficiency','Location','south');
                lgdd=legend([p5,p4,p6],'Tran.','Purity.','Eff.','Location','northwest');
                % title(lgdd,'TE_{00} to TE_{10}');
                % title(lgd,'TE_{10} to TE_{00}');
                else
                    p4=plot(real_wl, (real_purity),'*-r','LineWidth', 1);
                    hold on
                    p5=plot(real_wl, (real_tran),'*-b','LineWidth', 1);
                    p6=plot(real_wl, (real_tran.*real_purity),'-k','LineWidth', 1);
                    if noise==1
                        plot(real_wl, (real_noise),'--m','LineWidth', 1)
                    end
                % lgdd = legend({'Mode purity','Transmittance','Conversion efficiency'},'Location','north');
                % title(lgdd,'TE_{00} to TE_{10}');
                    % title("Tran: "+real_tran(center_idx)+", Purity: "+real_purity(center_idx)+", Eff: "+real_tran(center_idx)*real_purity(center_idx))
                end

                ylim([0.91 1.0])
                yticks([0.85, 0.9, 0.95, 1])
                yticklabels(["0.85" "0.9" "0.95" "1.0"])
                xlim([1.48 1.63])
                ylabel("Efficiency (a.u.)")
                xlabel("Wavelength (µm)")
        end

for output_load=0:1
            %% Load Output DFT
                for load_dft=0:0
                    %free space
                    Ex_o= ["Ex_te"+output_load+"0_field.h5"];                           %Hx
                    hinfo_Ex_o= hdf5info("Ex_te"+output_load+"0_field.h5");
                    Ey_o= ["Ey_te"+output_load+"0_field.h5"];                           %Hy
                    hinfo_Ey_o= hdf5info("Ey_te"+output_load+"0_field.h5");
                    Ez_o= ["Ez_te"+output_load+"0_field.h5"];                            %Ez
                    hinfo_Ez_o= hdf5info("Ez_te"+output_load+"0_field.h5");
                    
                    
                    Name_Exro=["/ex_"+0+".r"];
                    Name_Exio=["/ex_"+0+".i"];    
                    Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
                    Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
                    Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
                    % Hy
                    Name_Eyro=["/ey_"+0+".r"];
                    Name_Eyio=["/ey_"+0+".i"];    
                    Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
                    Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
                    Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
                    % Ez
                    Name_Ezro=["/ez_"+0+".r"];
                    Name_Ezio=["/ez_"+0+".i"];    
                    Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
                    Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
                    Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex
                                    
                    X0fig=100; Y0fig=100;
                    Wfig=1500;Hfig=900;
                    
                    MaxE=1;
                    
                    Maxa=max(Ex_data_ro,[],'all');
                    Maxb=max(Ey_data_ro,[],'all');
                    Maxc=max(Ez_data_ro,[],'all');
                    
                    MMax=[Maxa Maxb Maxc];
                    Ma=max(MMax);
                    Fsize=size(Ex_datao);
                    E_int_all=sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));
                    Max_int=max(E_int_all,[],'all');
                    X=Fsize(2);
                    Y=Fsize(1);
                    % for aspeact ratio
                    G=gcd(X,Y);
                    Xr=X/G;
                    Yr=Y/G;
                end


%% Input
            figure('Name','Results','position',[0 0 pt_2_cm*XY_w pt_2_cm*XY_h],'color','w')
                PC=pcolor(rot90(E_int_all,3)./Max_int);%intencity
                PC.EdgeColor='none';
                % title("Figure of Merit")
        
                hold on
                % c = colorbar();
                % c.Ticks=[] 
                % colormap("turbo");
                colormap('parula');
                caxis([0 1])
                
                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                clipY=round(Y*0.2);
                 set(gca,'YLim',[1 X])  
                 set(gca,'XLim',[1+clipY Y-clipY])
                pbaspect([Y*0.6 X 1]) %figure 종횡비
                % title("Input")
                axis off
                grid off


% 
%                 for load_dft=0:0
%                     %free space
%                     Ex_o= ["Ex_"+output_load+"_to_"+abs(output_load-1)+"_top.h5"];                           %Hx
%                     hinfo_Ex_o= hdf5info("Ex_"+output_load+"_to_"+abs(output_load-1)+"_top.h5");
%                     Ey_o= ["Ey_"+output_load+"_to_"+abs(output_load-1)+"_top.h5"];                           %Hy
%                     hinfo_Ey_o= hdf5info("Ey_"+output_load+"_to_"+abs(output_load-1)+"_top.h5");
%                     Ez_o= ["Ez_"+output_load+"_to_"+abs(output_load-1)+"_top.h5"];                            %Ez
%                     hinfo_Ez_o= hdf5info("Ez_"+output_load+"_to_"+abs(output_load-1)+"_top.h5");
% 
% 
%                     Name_Exro=["/ex_"+0+".r"];
%                     Name_Exio=["/ex_"+0+".i"];    
%                     Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
%                     Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
%                     Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
%                     % Hy
%                     Name_Eyro=["/ey_"+0+".r"];
%                     Name_Eyio=["/ey_"+0+".i"];    
%                     Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
%                     Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
%                     Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
%                     % Ez
%                     Name_Ezro=["/ez_"+0+".r"];
%                     Name_Ezio=["/ez_"+0+".i"];    
%                     Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
%                     Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
%                     Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex
% 
%                     X0fig=100; Y0fig=100;
%                     Wfig=1500;Hfig=900;
% 
%                     MaxE=1;
% 
%                     Maxa=max(Ex_data_ro,[],'all');
%                     Maxb=max(Ey_data_ro,[],'all');
%                     Maxc=max(Ez_data_ro,[],'all');
% 
%                     MMax=[Maxa Maxb Maxc];
%                     Ma=max(MMax);
%                     Fsize=size(Ex_datao);
%                     E_int_all=sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));
%                     Max_int=max(E_int_all,[],'all');
%                     X=Fsize(2);
%                     Y=Fsize(1);
%                     % for aspeact ratio
%                     G=gcd(X,Y);
%                     Xr=X/G;
%                     Yr=Y/G;
%                 end
% 
% 
%             figure('Name','Results','position',[0 0 pt_2_cm*8 pt_2_cm*3],'color','w')
%                 PC=pcolor(rot90(E_int_all,2)./Max_int);%intencity
%                 PC.EdgeColor='none';
% 
%                 hold on
%                 % c = colorbar();
%                 colormap("turbo");
%                 caxis([0 1])
% 
%                 set(gca,'fontname','arial','fontsize',10)
%                 set(gca,'linewidth',1);
%                  set(gca,'YLim',[1 Y])      
%                  set(gca,'XLim',[1 X])
%                 pbaspect([X Y 1]) %figure 종횡비
%                 % title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
%                 axis off
%                 grid off



















               for load_dft=0:0
                    %free space
                    Ex_o= ["Ex_"+output_load+"_to_"+abs(output_load-1)+"_field.h5"];                           %Hx
                    hinfo_Ex_o= hdf5info("Ex_"+output_load+"_to_"+abs(output_load-1)+"_field.h5");
                    Ey_o= ["Ey_"+output_load+"_to_"+abs(output_load-1)+"_field.h5"];                           %Hy
                    hinfo_Ey_o= hdf5info("Ey_"+output_load+"_to_"+abs(output_load-1)+"_field.h5");
                    Ez_o= ["Ez_"+output_load+"_to_"+abs(output_load-1)+"_field.h5"];                            %Ez
                    hinfo_Ez_o= hdf5info("Ez_"+output_load+"_to_"+abs(output_load-1)+"_field.h5");
                    
                    
                    Name_Exro=["/ex_"+0+".r"];
                    Name_Exio=["/ex_"+0+".i"];    
                    Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
                    Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
                    Ex_datao = Ex_data_ro + Ex_data_io * 1.0i;       %Complex
                    % Hy
                    Name_Eyro=["/ey_"+0+".r"];
                    Name_Eyio=["/ey_"+0+".i"];    
                    Ey_data_ro = h5read(Ey_o,Name_Eyro);          %Real
                    Ey_data_io = h5read(Ey_o,Name_Eyio);           %Imagine
                    Ey_datao = Ey_data_ro + Ey_data_io * 1.0i;       %Complex
                    % Ez
                    Name_Ezro=["/ez_"+0+".r"];
                    Name_Ezio=["/ez_"+0+".i"];    
                    Ez_data_ro = h5read(Ez_o,Name_Ezro);          %Real
                    Ez_data_io = h5read(Ez_o,Name_Ezio);           %Imagine
                    Ez_datao = Ez_data_ro + Ez_data_io * 1.0i;       %Complex
                    
                    MaxE=1;
                    
                    Maxa=max(Ex_data_ro,[],'all');
                    Maxb=max(Ey_data_ro,[],'all');
                    Maxc=max(Ez_data_ro,[],'all');
                    
                    MMax=[Maxa Maxb Maxc];
                    Ma=max(MMax);
                    Fsize=size(Ex_datao);
                    E_int_all=sqrt(Ex_datao.*conj(Ex_datao)+Ey_datao.*conj(Ey_datao)+Ez_datao.*conj(Ez_datao));
                    Max_int=max(E_int_all,[],'all');
                    X=Fsize(2);
                    Y=Fsize(1);
                    % for aspeact ratio
                    G=gcd(X,Y);
                    Xr=X/G;
                    Yr=Y/G;
               end


%% Output
           figure('Name','Results','position',[0 0 pt_2_cm*XY_w pt_2_cm*XY_h],'color','w')
                PC=pcolor(rot90(E_int_all,3)./Max_int);%intencity
                PC.EdgeColor='none';
        
                hold on
                % c = colorbar();
                % colormap("turbo");
                colormap('parula');
                caxis([0 1])
                
                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                clipY=round(Y*0.2);
                 set(gca,'YLim',[1 X])  
                 set(gca,'XLim',[1+clipY Y-clipY])
                pbaspect([Y*0.6 X 1]) %figure 종횡비
                % title("Output")
                axis off
                grid off


            real_tran=load("Real_flux_"+output_load+"_to_"+abs(output_load-1)+".txt");
            real_purity=load("Real_purity_"+output_load+"_to_"+abs(output_load-1)+".txt");
            if noise==1
                real_noise=load("Real_noise_"+output_load+"_to_"+abs(output_load-1)+".txt");
            end
            real_wl=1./load("Real_freqs_"+output_load+"_to_"+abs(output_load-1)+".txt");
            wl_var=abs(real_wl-center_wl);
            center_idx=find(wl_var==min(wl_var));

            figure('Name','Results','position',[500 100 pt_2_cm*5.0 pt_2_cm*3.8],'color','w')
                if output_load==1
                    plot(real_wl, (real_purity),'-r','LineWidth', 1)
                    hold on
                    plot(real_wl, (real_tran),'-b','LineWidth', 1)
                    plot(real_wl, (real_tran.*real_purity),'-k','LineWidth', 1)
                    if noise==1
                        plot(real_wl, (real_noise),'-m','LineWidth', 1)
                    end
                else
                    plot(real_wl, (real_purity),'-r','LineWidth', 1)
                    hold on
                    plot(real_wl, (real_tran),'-b','LineWidth', 1)
                    plot(real_wl, (real_tran.*real_purity),'-k','LineWidth', 1)
                    if noise==1
                        plot(real_wl, (real_noise),'--m','LineWidth', 1)
                    end
                    % title("Tran: "+real_tran(center_idx)+", Purity: "+real_purity(center_idx)+", Eff: "+real_tran(center_idx)*real_purity(center_idx))
                end
                ylim([0.89 1.01])
                yticks([0.9, 1])
                yticklabels(["0.9" "1.0"])
                xlim([1.48 1.63])
                ylabel("Efficiency")
                xlabel("Wavelength (µm)")
            T=real_tran(center_idx)
            P=real_purity(center_idx)
            E=real_tran(center_idx)*real_purity(center_idx)
            if noise==1
                N=real_noise(center_idx)
            end
            Nx=d_rs*d_height+1;
            Ny=d_rs*d_width+1;
            Nz=d_rs*d_length+1;


                for load_dft=0:0
                    %free space
                    Ex_o= ["Hx_"+output_load+"_to_"+abs(output_load-1)+"_top.h5"];                           %Hx
                    hinfo_Ex_o= hdf5info("Hx_"+output_load+"_to_"+abs(output_load-1)+"_top.h5");
                    
                    
                    Name_Exro=["/hx_"+0+".r"];
                    Name_Exio=["/hx_"+0+".i"];    
                    Ex_data_ro = h5read(Ex_o,Name_Exro);          %Real
                    Ex_data_io = h5read(Ex_o,Name_Exio);           %Imagine
                    Ex_datao = Ex_data_ro;% + Ex_data_io * 1.0i;       %Complex

                    
                    MaxE=1;
                    
                    Maxa=max(Ex_data_ro,[],'all');
                    
                    MMax=[Maxa Maxb Maxc];
                    Ma=max(MMax);
                    Fsize=size(Ex_datao);
                    E_int_all=(Ex_datao.*conj(Ex_datao));
                    Max_int=max(E_int_all,[],'all');
                    X=Fsize(2);
                    Y=Fsize(1);
                    % for aspeact ratio
                    G=gcd(X,Y);
                    Xr=X/G;
                    Yr=Y/G;
                end

            figure('Name','Results','position',[500 100 pt_2_cm*16.5 pt_2_cm*4.2],'color','w')
                PC=pcolor(rot90(E_int_all,1)./Max_int);%intencity
                PC.EdgeColor='none';

                hold on
                % c = colorbar();
                colormap("hot");
                caxis([0 1])

                set(gca,'fontname','arial','fontsize',10)
                set(gca,'linewidth',1);
                 set(gca,'YLim',[1 X])      
                 set(gca,'XLim',[1 Y])
                pbaspect([Y X 1]) %figure 종횡비
                % title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
                axis off
                grid off

            % figure('Name','Results','position',[500 100 pt_2_cm*5.2 pt_2_cm*12],'color','w')
            %     PC=pcolor(rot90(E_int_all,2)./Max_int);%intencity
            %     PC.EdgeColor='none';
            % 
            %     hold on
            %     % c = colorbar();
            %     colormap("turbo");
            %     caxis([0 1])
            % 
            %     set(gca,'fontname','arial','fontsize',10)
            %     set(gca,'linewidth',1);
            %      set(gca,'YLim',[1 Y])      
            %      set(gca,'XLim',[1 X])
            %     pbaspect([X Y 1]) %figure 종횡비
            %     % title("\surd{|E_x|^2+|E_y|^2+|E_z|^2}")
            %     axis off
            %     grid off


    end
            % % Geometry=load('modified_design.txt');
            % Geometry=load('Binarized_iter94_beta128.txt');
            % Gs=size(Geometry);
            % My_width= linspace(-d_width/2, d_width/2, Ny);
            % My_length= linspace(0, d_length, Nz);
            % My_height= linspace(0, d_height, Nx);
            % 
            % Geometry_new=(reshape(Geometry,Nz,Ny,Nx)); % 
            % X_coordinate=Nx;
            % figure('Name','Results','position',[0 0 310/0.7 210/0.9],'color','w')  
            %     PC2=pcolor(My_width, My_length, rot90(Geometry_new(:,:,X_coordinate),2));
            %     PC2.EdgeColor='none';   
            %     hold on   
            %     colormap(flipud(gray));
            %     c = colorbar();
            %     caxis([0 1])
            %     set(gca,'fontname','arial','fontsize',10)
            %     set(gca,'linewidth',1);
            %     pbaspect([Ny Nz 1]) %figure 종횡비
            %     title('X=top')
            %     view(2);% = view(0,90) < 방위각, 고도각
            % 
            % X_coordinate=round(Nx/2);
            % figure('Name','Results','position',[0 0 310/0.7 210/0.9],'color','w')
            %     PC2=pcolor(My_width, My_length, rot90(Geometry_new(:,:,X_coordinate),2));
            %     PC2.EdgeColor='none';   
            %     hold on   
            %     colormap(flipud(gray));
            %     c = colorbar();
            %     caxis([0 1])
            %     set(gca,'fontname','arial','fontsize',10)
            %     set(gca,'linewidth',1);
            %     pbaspect([Ny Nz 1]) %figure 종횡비
            %     title('X=middle')
            %     view(2);% = view(0,90) < 방위각, 고도각
            % 
            % X_coordinate=1;
            % figure('Name','Results','position',[0 0 310/0.7 210/0.9],'color','w')
            %     PC2=pcolor(My_width, My_length, rot90(Geometry_new(:,:,X_coordinate),2));
            %     PC2.EdgeColor='none';   
            %     hold on   
            %     colormap(flipud(gray));
            %     c = colorbar();
            %     caxis([0 1])
            %     set(gca,'fontname','arial','fontsize',10)
            %     set(gca,'linewidth',1);
            %     pbaspect([Ny Nz 1]) %figure 종횡비
            %     title('X=bottom')
            %     view(2);% = view(0,90) < 방위각, 고도각



            