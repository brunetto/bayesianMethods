model in diffpriors.bug
data in diffpriors.dat
compile, nchains(3) 			
initialize 						
update 1000 					
monitor set s, thin(10) 		
update 10000 					
coda * 							
data to useddata					

