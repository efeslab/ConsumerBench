from applications.SleepApplication import SleepApplication
from applications.AnotherApplication import AnotherApplication
from workflow import Workflow
import globals

def main():
    """Test complex workflow with dependencies"""
    
    print("=== Testing Complex Workflow with Dependencies ===\n")
    
    # Initialize globals
    globals.set_start_time()
    globals.set_results_dir("results")
    
    # Create application instances
    sleepApplication1 = SleepApplication()
    anotherApplication = AnotherApplication()
    
    # Create workflow from YAML
    workflow = Workflow("configs/test_complex.yml")
    
    # Register applications
    workflow.register_application("SleepApplication", sleepApplication1)
    workflow.register_application("AnotherApplication", anotherApplication)
    
    print("Registered applications:")
    for app_name, app in workflow.applications.items():
        print(f"  - {app_name}: {type(app).__name__}")
    print()
    
    # Load workflow configuration
    workflow.load_workflow_unit_config()
    print("Loaded workflow configuration:")
    for unit_name, unit_config in workflow.workflow_unit_map.items():
        print(f"  - {unit_name}: {unit_config['type']} (count: {unit_config['count']})")
        print(f"    Config: {unit_config['node_config']}")
    print()
    
    # Generate task queue
    workflow.generate_task_queue()
    print("Generated task queue:")
    for k, v in workflow.tasks_map_queue.items():
        print(f"Task group {k}:")
        for unit in v:
            print(f"  - {unit.type} (ID: {unit.id})")
            print(f"    Start node: {unit.node_start}")
            print(f"    End node: {unit.node_end}")
    print()
    
    # Generate benchmark
    bm = workflow.generate_benchmark()
    print("Benchmark generated successfully.")
    
    # Visualize the benchmark
    bm.visualize("complex_workflow_benchmark.png")
    print("Benchmark visualization saved to 'complex_workflow_benchmark.png'")
    
    # Run the benchmark
    print("\n=== Running Benchmark ===")
    total_time = bm.run_concurrent()
    print(f"Total execution time: {total_time:.4f} seconds")
    
    # Display results
    print("\n=== Results ===")
    bm.display_results()

if __name__ == "__main__":
    main() 