import time

class Timer:
    def __init__(self, show_progress = False, iterations =  None):
        self.global_start_time = time.time()
        self.local_start_time_dict = { 0: self.global_start_time}
        self.local_time_elapsed_dict = {}
        self.local_time_elapsed_list = []
        self.local_start_time_index = 1
        self.show_progress = show_progress
        self.total_iterations = iterations
        
        if self.show_progress == True and self.total_iterations is not None:
            print(f"Processing --> {self.local_start_time_index}/{self.total_iterations}")
    
    def get_global_time_elapsed(self):
        """
        Returns: Total time elapsed in minutes
        """
        
        global_time_elapsed = round((time.time()-self.global_start_time)/60 , 5)
        
        print(f"Total time taken: {global_time_elapsed} mins")
        return({'time_elapsed_mins':global_time_elapsed})
    
    def update(self):
        
        clear_output()
        
        current_time = time.time()
        
        local_time_elapsed = round(
            (current_time - self.local_start_time_dict[self.local_start_time_index-1])/60,
            2)
        
        self.local_time_elapsed_dict[self.local_start_time_index-1] = local_time_elapsed
        self.local_time_elapsed_list.append(local_time_elapsed)
        
        if (np.sum(np.array(self.local_time_elapsed_list))) == 0:
            iter_per_min = '--'
        else:
            iter_per_min = round(self.local_start_time_index/(np.sum(np.array(self.local_time_elapsed_list))), 2)
        
        
        self.local_start_time_dict[self.local_start_time_index] = current_time
        
        if self.show_progress == True and self.total_iterations == None:
            print(f"Process {self.local_start_time_index} completed in {self.local_time_elapsed_dict[self.local_start_time_index-1]} mins ")
            print(f"Total time elapsed {round((time.time()-self.global_start_time)/60,2)} mins")
            print(f"Speed --> {iter_per_min} iter/min")
        
        if self.show_progress == True and self.total_iterations is not None:
            if iter_per_min == '--':
                estimated_complete_time = '--'
            else:
                estimated_complete_time = round((1/iter_per_min)*(self.total_iterations-self.local_start_time_index), 2)
            
            print(f"Processing : {self.local_start_time_index+1}/{self.total_iterations}")
            #print(f"Process {self.local_start_time_index} completed in {self.local_time_elapsed_dict[self.local_start_time_index-1]} mins ")
            print(f"Speed : {iter_per_min} iter/min")
            print(f"Estimated Completion Time : {estimated_complete_time} mins")
        
    
        self.local_start_time_index += 1
