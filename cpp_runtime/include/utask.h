#include <string>

#define NO_OVERLAP_TASK 0       // No overlap between input and output arrays
#define OVERLAP_TASK 1          // Overlap between input and output
#define LOCAL_TASK 2            // No input dependency

namespace FluidGPU{
    class uTask {
    public:
        void** input_arrays;
        void** output_arrays;
        void** local_arrays;
        int* input_size;
        int* output_size;
        int local_array_num;
        int input_num;
        int output_num;
        unsigned task_id;
        unsigned task_rank;
        std::string task_name;
        double task_time;

    public:

        uTask(void** input_arrays, int* input_size, int input_num,
              void** output_arrays, int* output_size, int output_num,
              void** local_arrays, int local_array_num,
              unsigned task_id, unsigned task_rank, std::string task_name, double task_time = 0.0)
            : input_arrays(input_arrays), output_arrays(output_arrays),
              input_size(input_size), output_size(output_size),
              local_arrays(local_arrays), local_array_num(local_array_num),
              input_num(input_num), output_num(output_num),
              task_id(task_id), task_rank(task_rank), task_name(task_name), task_time(task_time) {
        }

        virtual ~uTask() = default;

        uTask(const uTask& other) {
            input_num = other.input_num;
            output_num = other.output_num;
            local_array_num = other.local_array_num;
            task_id = other.task_id;
            task_rank = other.task_rank;
            task_name = other.task_name;
            task_time = other.task_time;


            input_arrays = new void*[input_num];
            for (int i = 0; i < input_num; ++i) {
                input_arrays[i] = other.input_arrays[i];
            }

            output_arrays = new void*[output_num];
            for (int i = 0; i < output_num; ++i) {
                output_arrays[i] = other.output_arrays[i];
            }

            local_arrays = new void*[local_array_num];
            for (int i = 0; i < local_array_num; ++i) {
                local_arrays[i] = other.local_arrays[i];    
            }

            input_size = new int[input_num];
            for (int i = 0; i < input_num; ++i) {
                input_size[i] = other.input_size[i];
            }

            output_size = new int[output_num];
            for (int i = 0; i < output_num; ++i) {
                output_size[i] = other.output_size[i];
            }
        }

        uTask& operator=(const uTask& other) {
            if (this == &other) return *this;

            input_num = other.input_num;
            output_num = other.output_num;
            local_array_num = other.local_array_num;
            task_id = other.task_id;
            task_rank = other.task_rank;
            task_name = other.task_name;
            task_time = other.task_time;

            input_arrays = new void*[input_num];
            for (int i = 0; i < input_num; ++i) {
                input_arrays[i] = other.input_arrays[i];
            }

            output_arrays = new void*[output_num];
            for (int i = 0; i < output_num; ++i) {
                output_arrays[i] = other.output_arrays[i];
            }
            local_arrays = new void*[local_array_num];
            for (int i = 0; i < local_array_num; ++i) {
                local_arrays[i] = other.local_arrays[i];
            }
            input_size = new int[input_num];
            for (int i = 0; i < input_num; ++i) {
                input_size[i] = other.input_size[i];
            }
            output_size = new int[output_num];
            for (int i = 0; i < output_num; ++i) {
                output_size[i] = other.output_size[i];
            }
            return *this;
        }

        static int bufferTypeCheck(uTask& task) 
        {
            if (task.input_num == 0 && task.output_num == 0) {
                return LOCAL_TASK;
            }
            for (int i = 0; i < task.input_num; ++i) {
                void* in_ptr = task.input_arrays[i];
                for (int j = 0; j < task.output_num; ++j) {
                    if (in_ptr == task.output_arrays[j]) {
                        return OVERLAP_TASK;
                    }
                }
            }
            return NO_OVERLAP_TASK; 
        }
    };
}