__kernel void Kernel(__global float *Cin,                        
            __global float *weight,                        
            __global float *C)                        
{                                                   
   


int i = get_group_id(2);
int i1=get_group_id(1);
int i2=get_local_id(0);
__local float weight_local[5][5];
__local float C_local[8][112];
__local float Cin_local[12][116];


for(int dd=0; dd<224; dd+=112){


barrier(CLK_LOCAL_MEM_FENCE);
 for (int w = 0; w < 8; w++) 
  C_local[w][i2]=C[i * 50176 + (i1*8+w) * 224 + i2+dd];
barrier(CLK_LOCAL_MEM_FENCE); 


      for (int j = 0; j < 256; j+=1) {



//load 
barrier(CLK_LOCAL_MEM_FENCE);

                            if(i2<25){
                            int hang=i2/5;
                            int lie=i2%5;
                            weight_local[hang][lie] = weight[i * 6400 + j * 25 + hang * 5 + lie];
                            }
       for (int shu=0; shu<12; shu++){
       Cin_local[shu][i2]=Cin[j*51984+(i1*8+shu)*228+i2+dd];
       if(i2<4)
       Cin_local[shu][i2+112]=Cin[j*51984+(i1*8+shu)*228+i2+dd+112];
}
barrier(CLK_LOCAL_MEM_FENCE);

            

	 for (int w = 0; w < 8; w++) {
barrier(CLK_LOCAL_MEM_FENCE);
           float temp=C_local[w][i2];
	  	   for (int pp = 0; pp < 5; pp++) 
	   	     for (int qq = 0; qq < 5; qq++) 
                temp+= Cin_local[w+pp][i2+qq]*weight_local[pp][qq];
           C_local[w][i2]=temp; 
barrier(CLK_LOCAL_MEM_FENCE);   
             }
   }


for (int w = 0; w < 8; w++) {
  if(C_local[w][i2]<0) C_local[w][i2]=0;
  C[i * 50176 + (i1*8+w) * 224 + i2+dd] = C_local[w][i2];
}

}


}                                                   









