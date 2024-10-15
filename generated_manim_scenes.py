
from manim import *

# Existing Scene classes and functions

# Insert generated Scene classes below


# Generated Scene Classes
class SpaceIsALatentSequenceATheoryOfTheHippocampusScene(Scene):
    def construct(self):
        # Title
        title = Text("Space is a latent sequence: A theory of the hippocampus").scale(0.8).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Content explanation
        content = (
            "This article presents a unifying theory that the hippocampus represents space as an emergent property of latent higher-order sequence learning."
        )
        content_text = Text(content, font_size=24).scale(0.5).next_to(title, DOWN, buff=0.5)
        self.play(Write(content_text))
        self.wait(3)

        # Introduce the CSCG model
        cscg_model = Text("Clone-Structured Causal Graph (CSCG) Model", font_size=28).next_to(content_text, DOWN, buff=0.5)
        self.play(Write(cscg_model))
        self.wait(2)

        # Explanation of the model
        explain_model = (
            "Explains various phenomena observed in hippocampal function by treating spatial representations as sequences rather than Euclidean mappings."
        )
        explain_text = Text(explain_model, font_size=24).scale(0.5).next_to(cscg_model, DOWN, buff=0.5)
        self.play(Write(explain_text))
        self.wait(3)

        # Visual Representation of Sequential Learning
        sequence_graph = SelfContainedGraph()
        self.play(FadeOut(content_text), FadeOut(cscg_model), FadeOut(explain_text))
        self.play(Create(sequence_graph))
        self.wait(3)

        # Emphasizing Allocentric Cognitive Maps
        allocentric_text = Text("Allocentric Cognitive Maps for Navigation and Planning", font_size=28).next_to(sequence_graph, DOWN, buff=0.5)
        self.play(Write(allocentric_text))
        self.wait(2)

        # End Scene
        self.play(FadeOut(allocentric_text), FadeOut(sequence_graph), FadeOut(title))
        self.wait(1)

class SelfContainedGraph(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_graph()

    def create_graph(self):
        # Create a simple graph structure to represent sequences
        points = [Dot(point=RIGHT * x + UP * 2) for x in range(-3, 4)]
        lines = [Line(start=points[i].get_center(), end=points[i + 1].get_center()) for i in range(len(points) - 1)]

        for line in lines:
            self.add(line)

        for point in points:
            self.add(point)

class CloneStructuredCognitiveGraphsForLearningContextSpecificRepresentationsScene(Scene):
    def construct(self):
        # Title
        title = Text("Clone-structured Cognitive Graphs", font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Introduce the concept of challenges in learning cognitive maps
        challenges = Text("Challenges in Learning Cognitive Maps", font_size=24).next_to(title, DOWN)
        self.play(Write(challenges))
        self.wait(1)

        # Visual representation of sensory observations and latent space
        observations = Circle(radius=0.5, color=BLUE).shift(LEFT * 3)
        latent_space = Circle(radius=0.5, color=GREEN).shift(RIGHT * 3)
        self.play(Create(observations), Create(latent_space))
        
        # Connect observations to latent space
        self.play(Create(Line(observations.get_right(), latent_space.get_left(), color=WHITE)))
        self.wait(1)

        # Introduce the Clone-structured Cognitive Graph (CSCG) model
        cscg_title = Text("Clone-structured Cognitive Graph (CSCG)", font_size=24).next_to(challenges, DOWN)
        self.play(Write(cscg_title))
        self.wait(1)

        # Visualize the cloning structure
        clone_structure = VGroup(
            Dot(color=YELLOW).shift(LEFT * 2 + DOWN),
            Dot(color=YELLOW).shift(RIGHT * 2 + DOWN),
            Line(LEFT * 2 + DOWN, RIGHT * 2 + DOWN, color=YELLOW)
        )
        self.play(Create(clone_structure))
        self.wait(1)

        # Show merging and splitting of contexts
        merge_split_text = Text("Merging and Splitting Contexts", font_size=20).next_to(cscg_title, DOWN)
        self.play(Write(merge_split_text))
        
        merge_line = Line(clone_structure[0].get_center(), clone_structure[1].get_center(), color=RED)
        self.play(Create(merge_line))
        self.wait(1)

        # Introduce action-conditional probabilities
        action_conditional = Text("Action-conditional Probabilities", font_size=20).next_to(merge_split_text, DOWN)
        self.play(Write(action_conditional))
        self.wait(1)

        # Show expectation maximization
        expectation_text = Text("Expectation Maximization Algorithm", font_size=20).next_to(action_conditional, DOWN)
        self.play(Write(expectation_text))
        self.wait(1)

        # Visualize latent topologies
        latent_topology = Polygon(np.array([-2, -2, 0]), np.array([2, -2, 0]), np.array([0, 2, 0]), color=PURPLE, fill_opacity=0.5)
        self.play(Create(latent_topology))
        self.wait(1)

        # Conclude with improved representations
        conclusion = Text("Improved Representation of Observed Sequences", font_size=24).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)

        # Fade out all elements
        self.play(FadeOut(VGroup(title, challenges, observations, latent_space, cscg_title, clone_structure, merge_split_text, action_conditional, expectation_text, latent_topology, conclusion)))

class CscgModelAndItsApplicationsInLearningAndNavigationScene(Scene):
    def construct(self):
        # Title
        title = Text("CSCG Model and Its Applications in Learning and Navigation")
        title.scale(0.8)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Content introduction
        content_intro = Text("Investigating CSCG in learning latent topologies")
        content_intro.next_to(title, DOWN, buff=0.5)
        self.play(Write(content_intro))
        self.wait(1)

        # Create a representation of environments
        environment = Circle(radius=1.5, color=BLUE).shift(LEFT * 3)
        environment_label = Text("Environment 1").next_to(environment, UP)
        self.play(Create(environment), Write(environment_label))
        
        # Represent perceptually aliased observations
        observations = Square(side_length=1.5, color=GREEN).shift(RIGHT * 3)
        observations_label = Text("Perceptually Aliased Observations").next_to(observations, UP)
        self.play(Create(observations), Write(observations_label))

        # Transition to show multiple environments
        self.wait(1)
        self.play(FadeOut(content_intro), FadeOut(environment_label), FadeOut(observations_label))
        
        # Add multiple environments
        for i in range(2, 5):
            new_env = Circle(radius=1.5, color=BLUE).shift(LEFT * (3 + (i - 2) * 2))
            new_env_label = Text(f"Environment {i}").next_to(new_env, UP)
            self.play(Create(new_env), Write(new_env_label))
            self.wait(0.5)

        # Transition to stitching global maps
        self.play(FadeOut(*self.mobjects), run_time=1)
        stitching_title = Text("Stitching Global Maps from Disjoint Experiences")
        stitching_title.scale(0.7)
        self.play(Write(stitching_title))
        self.wait(1)

        # Show the stitching process with arrows
        arrow1 = Arrow(start=LEFT * 3, end=RIGHT * 3, buff=0)
        arrow2 = Arrow(start=LEFT * 3, end=RIGHT * 3, buff=0).shift(DOWN * 1.5)
        self.play(Create(arrow1), Create(arrow2))
        self.wait(1)

        # Explain hippocampal phenomena
        phenomena_title = Text("Explaining Hippocampal Phenomena").next_to(stitching_title, DOWN, buff=0.5)
        self.play(Write(phenomena_title))
        self.wait(1)

        # Final transition to conclusions
        conclusion_title = Text("CSCG: Understanding Animal Navigation and Memory")
        conclusion_title.scale(0.8)
        conclusion_title.next_to(phenomena_title, DOWN, buff=0.5)
        self.play(Write(conclusion_title))
        self.wait(2)

        # Fade out for end
        self.play(FadeOut(stitching_title), FadeOut(phenomena_title), FadeOut(conclusion_title))
        self.wait(1)

class CscgsLearningAndNavigatingInComplexEnvironmentsScene(Scene):
    def construct(self):
        # Title
        title = Text("CSCGs: Learning and Navigating in Complex Environments", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        
        # Content details
        content = [
            "CSCGs utilize learned transition graphs as schemas.",
            "Agents trained in known environments adapt to unfamiliar spaces.",
            "Emission matrix updates while keeping transition matrix fixed.",
            "CSCGs account for changes in geometry and visual cues.",
            "Sequential context interpretation reproduces place field remapping."
        ]
        
        # Create a list of text objects for content
        content_texts = [Text(text, font_size=24) for text in content]
        for i, text in enumerate(content_texts):
            text.next_to(title, DOWN, buff=0.5 + i*0.3)
            self.play(Write(text))
            self.wait(0.5)
        
        # Transition to a visual representation of the CSCG
        self.play(FadeOut(title), *[FadeOut(text) for text in content_texts])
        
        # Create a graph representation of a CSCG
        nodes = [Dot(point=RIGHT * i) for i in range(-3, 4)]
        edges = []
        for i in range(len(nodes) - 1):
            edge = Line(nodes[i].get_center(), nodes[i + 1].get_center())
            edges.append(edge)
        
        # Adding nodes and edges to the scene
        self.play(*[Create(node) for node in nodes])
        self.play(*[Create(edge) for edge in edges])
        self.wait(1)
        
        # Show transition and emission matrices
        transition_matrix = Matrix([[0.8, 0.2], [0.1, 0.9]], left_bracket='[', right_bracket=']')
        emission_matrix = Matrix([[0.6, 0.4], [0.3, 0.7]], left_bracket='[', right_bracket=']')
        
        transition_matrix.to_edge(LEFT)
        emission_matrix.to_edge(RIGHT)
        
        self.play(Write(transition_matrix), Write(emission_matrix))
        self.wait(1)
        
        # Update emission matrix
        updated_emission_matrix = Matrix([[0.7, 0.3], [0.4, 0.6]], left_bracket='[', right_bracket=']')
        self.play(Transform(emission_matrix, updated_emission_matrix))
        self.wait(1)
        
        # Final explanation
        final_explanation = Text("CSCGs adapt and navigate effectively.", font_size=28)
        final_explanation.to_edge(DOWN)
        self.play(Write(final_explanation))
        self.wait(2)
        
        # Ending the scene
        self.play(FadeOut(final_explanation), FadeOut(transition_matrix), FadeOut(emission_matrix), *[FadeOut(node) for node in nodes], *[FadeOut(edge) for edge in edges])
        self.wait(1)

class ExplainingPlaceFieldRemappingThroughSequentialContextsScene(Scene):
    def construct(self):
        # Title
        title = Text("Explaining Place Field Remapping through Sequential Contexts", font_size=24)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Introduction to Place Field Remapping
        intro = Text("Place Field Remapping in Neural Responses", font_size=20)
        self.play(Write(intro))
        self.wait(2)
        self.play(FadeOut(intro))

        # Sequence Representation vs. Geometric Models
        sequence_model = Text("Sequence Representation vs. Geometric Models", font_size=20)
        self.play(Write(sequence_model))
        self.wait(2)

        # Create a diagram of traditional geometric models (rectangles)
        geom_model = Rectangle(height=1, width=2, color=BLUE).shift(LEFT)
        seq_model = Circle(radius=1, color=GREEN).shift(RIGHT)
        self.play(Create(geom_model), Create(seq_model))
        self.wait(2)

        # Label the models
        geom_label = Text("Geometric Model", font_size=16).next_to(geom_model, DOWN)
        seq_label = Text("Sequence Representation", font_size=16).next_to(seq_model, DOWN)
        self.play(Write(geom_label), Write(seq_label))
        self.wait(2)

        # Transition to environmental changes
        self.play(FadeOut(geom_model), FadeOut(seq_model), FadeOut(geom_label), FadeOut(seq_label))
        context_change = Text("Adapting to Environmental Changes", font_size=20)
        self.play(Write(context_change))
        self.wait(2)

        # Show a simple environment with landmarks (using dots)
        landmarks = VGroup(*[Dot(point=RIGHT * x) for x in range(-3, 4)])
        self.play(Create(landmarks))
        self.wait(2)

        # Show place fields adapting to local changes
        place_fields = VGroup(*[Circle(radius=0.5, color=YELLOW).move_to(landmark.get_center()) for landmark in landmarks])
        self.play(FadeIn(place_fields))
        self.wait(2)

        # Transition to global representation
        global_rep = Text("Global Representation of Location", font_size=20)
        self.play(Transform(context_change, global_rep))
        self.wait(2)

        # Show a larger circle representing global representation
        global_circle = Circle(radius=3, color=RED, stroke_width=4).move_to(ORIGIN)
        self.play(Create(global_circle))
        self.wait(2)

        # Final thoughts
        final_thoughts = Text("Place Cells: Encoding Spatial & Temporal Information", font_size=20)
        self.play(Write(final_thoughts))
        self.wait(3)

        # Fade out everything
        self.play(FadeOut(final_thoughts), FadeOut(global_circle), FadeOut(landmarks), FadeOut(place_fields))
        self.wait(2)

class MechanismsOfPlaceFieldDynamicsInCscgsScene(Scene):
    def construct(self):
        self.create_title()
        self.create_content()
        self.show_transitions()
        self.finalize_scene()

    def create_title(self):
        title = Text("Mechanisms of Place Field Dynamics in CSCGs", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

    def create_content(self):
        content = [
            "Investigates Contextual Sequential Graphs (CSCGs)",
            "Focuses on place field dynamics",
            "Place fields remain consistent despite environmental changes",
            "Visual cues influence place field stability",
            "Explains place field repetition and size variations",
            "Highlights uniqueness of visual context in remapping"
        ]
        
        content_mobjects = VGroup(*[Text(text, font_size=24) for text in content]).arrange(DOWN, buff=0.5)
        self.play(LaggedStartMap(Write, content_mobjects, lag_ratio=0.5))
        self.wait(2)

    def show_transitions(self):
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=1)
        
        transition_title = Text("Key Concepts", font_size=36)
        transition_title.to_edge(UP)
        self.play(Write(transition_title))
        
        concepts = [
            "Place Field Dynamics",
            "Remapping Mechanisms",
            "Visual Context Influence"
        ]
        
        concepts_mobjects = VGroup(*[Text(concept, font_size=24) for concept in concepts]).arrange(DOWN, buff=0.5)
        self.play(LaggedStartMap(Write, concepts_mobjects, lag_ratio=0.5))
        self.wait(2)

    def finalize_scene(self):
        final_message = Text("Understanding the mechanisms behind neural responses", font_size=24)
        final_message.move_to(DOWN)
        self.play(Write(final_message))
        self.wait(2)
        self.play(FadeOut(final_message), run_time=1)

class CscgModelAndPlaceCellRepresentationsScene(Scene):
    def construct(self):
        title = Text("CSCG Model and Place Cell Representations", font_size=36)
        self.play(Write(title))
        self.wait(1)

        content = (
            "The CSCG model provides a framework for understanding how place cells in the hippocampus "
            "represent spatial information through sequential contexts. It highlights the complexities of "
            "place field mapping, demonstrating that place fields can change based on environmental modifications "
            "and local cues."
        )
        content_text = Text(content, font_size=24, line_spacing=0.5).scale(0.7)
        content_text.move_to(ORIGIN + DOWN)
        self.play(Write(content_text))
        self.wait(3)

        transition_text = Text("Contrasts with traditional views", font_size=24, color=YELLOW).scale(0.7)
        transition_text.move_to(UP * 2)
        self.play(Transform(content_text, transition_text))
        self.wait(2)

        insights_text = Text("Spatial representation: sequences over fixed locations", font_size=24, color=BLUE).scale(0.7)
        insights_text.move_to(DOWN * 2)
        self.play(Transform(transition_text, insights_text))
        self.wait(3)

        final_insight = Text("Insights relevant to cognitive science and AI", font_size=24, color=GREEN).scale(0.7)
        self.play(Transform(insights_text, final_insight))
        self.wait(3)

        self.play(FadeOut(title), FadeOut(final_insight))
        self.wait(1)

class ExploringClonedStateConditionalGraphsCscgInSpatialRepresentationLearningScene(Scene):
    def construct(self):
        # Title
        title = Text("Exploring Cloned State-Conditional Graphs (CSCG)", font_size=36)
        subtitle = Text("in Spatial Representation Learning", font_size=24)
        self.play(Write(title), Write(subtitle.next_to(title, DOWN)))
        self.wait(2)
        self.clear()

        # Content Visualization
        cscg_text = Text("CSCG: Dynamic Environments & Multiple Maps", font_size=28, color=BLUE)
        tem_text = Text("TEM: Limitations in Learning", font_size=28, color=RED)
        self.play(Write(cscg_text))
        self.wait(1)
        self.play(Write(tem_text.next_to(cscg_text, DOWN)))
        self.wait(2)
        self.clear()

        # Comparison of CSCG and TEM
        cscg_rect = Rectangle(width=3, height=1, color=BLUE).shift(LEFT * 3)
        tem_rect = Rectangle(width=3, height=1, color=RED).shift(RIGHT * 3)
        self.play(Create(cscg_rect), Create(tem_rect))
        
        cscg_label = Text("CSCG", font_size=24).move_to(cscg_rect.get_center())
        tem_label = Text("TEM", font_size=24).move_to(tem_rect.get_center())
        self.play(Write(cscg_label), Write(tem_label))
        self.wait(2)
        self.clear()

        # Highlighting the arguments
        argument_text = Text("Hippocampal Phenomena: Artifacts of Euclidean Mapping", font_size=24)
        self.play(Write(argument_text))
        self.wait(2)
        self.clear()

        # Future Research Directions
        future_text = Text("Future Directions: Active Learning & Schemas", font_size=28, color=GREEN)
        self.play(Write(future_text))
        self.wait(3)
        
        # Fade out for ending
        self.play(FadeOut(future_text))

class LearningClonedHiddenMarkovModelsAndActionAugmentedClonedHmmsScene(Scene):
    def construct(self):
        # Title
        title = Text("Learning Cloned Hidden Markov Models and Action-Augmented Cloned HMMs", font_size=24)
        title.to_edge(UP)

        # Content elements
        content = [
            "Expectation-Maximization (EM) Learning",
            "Baum-Welch Algorithm",
            "Optimizes Transition Matrix & Prior Probabilities",
            "Exploits Sparsity for Efficiency",
            "Conditional State Cloned Graphs (CSCGs)",
            "Incorporates Actions at Each Time Step",
            "Learning for Discrete and Continuous Observations",
            "Transfer Learning Capabilities"
        ]

        # Create content text objects
        content_texts = VGroup(*[Text(text, font_size=20) for text in content])
        content_texts.arrange(DOWN, buff=0.5).next_to(title, DOWN, buff=0.5)

        # Add title and content to the scene
        self.play(Write(title))
        self.play(LaggedStartMap(Write, content_texts))
        self.wait(1)

        # Transition to a visual representation of HMM
        self.clear()
        self.play(FadeIn(title))

        # Create a simple state transition diagram for HMM
        states = [Circle(radius=0.3).shift(LEFT * 2 + UP * i) for i in range(-1, 2)]
        arrows = [Arrow(states[i].get_right(), states[i+1].get_left(), buff=0.1) for i in range(len(states)-1)]
        state_labels = [Text(f"S{i+1}", font_size=20).move_to(states[i].get_center()) for i in range(len(states))]

        # Add states and labels to the scene
        self.play(LaggedStart(*[FadeIn(state) for state in states]))
        self.play(LaggedStart(*[Write(label) for label in state_labels]))
        self.play(LaggedStartMap(ShowCreation, arrows))
        self.wait(1)

        # Highlight the transition optimization
        transition_text = Text("Optimizing Transition Matrix", font_size=20).next_to(title, DOWN)
        self.play(Transform(title, transition_text))
        self.wait(1)
        
        # Clear and introduce CSCGs
        self.clear()
        self.play(FadeIn(transition_text))

        # Create graph representation of CSCGs
        actions = [Circle(radius=0.2).shift(RIGHT * i) for i in range(3)]
        action_labels = [Text(f"A{i+1}", font_size=18).move_to(actions[i].get_center()) for i in range(len(actions))]
        
        # Add actions to the scene
        self.play(LaggedStart(*[FadeIn(action) for action in actions]))
        self.play(LaggedStart(*[Write(label) for label in action_labels]))
        self.wait(1)

        # Summarize learning process
        summary_text = Text("Learning Process: Discrete & Continuous Observations", font_size=20).shift(DOWN * 2)
        self.play(Write(summary_text))
        self.wait(2)

        # Final fade out
        self.play(FadeOut(VGroup(title, content_texts, transition_text, summary_text, *states, *actions, *arrows)))
        self.wait(1)

class LearningProcedureForContinuousStateConditionalGaussianMixtureModelsScene(Scene):
    def construct(self):
        self.create_title()
        self.create_steps()
        self.create_vector_quantization()
        self.create_transition_matrix()
        self.create_viterbi_training()
        self.create_noisy_observations()
        self.create_place_fields()
        self.finalize_scene()

    def create_title(self):
        title = Text("Learning Procedure for Continuous State-Conditional Gaussian Mixture Models", font_size=24)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

    def create_steps(self):
        steps = VGroup(
            Text("Step 1: Fix Transition Matrix", font_size=20),
            Text("Optimize Means using K-means Clustering", font_size=20),
            Text("Step 2: Optimize Transition Matrix", font_size=20),
            Text("Using Expectation-Maximization (EM) Algorithm", font_size=20)
        ).arrange(DOWN, buff=0.5)
        steps.move_to(ORIGIN)
        self.play(Write(steps))
        self.wait(2)
        self.play(FadeOut(steps))

    def create_vector_quantization(self):
        quantization = Text("Vector Quantization of Data", font_size=20)
        quantization.move_to(UP * 2)
        self.play(Write(quantization))
        self.wait(2)
        self.play(FadeOut(quantization))

    def create_transition_matrix(self):
        transition_matrix = Tex(
            r"Transition Matrix: \quad P(s_t | s_{t-1}, a_t)", font_size=20
        )
        transition_matrix.move_to(UP * 2)
        self.play(Write(transition_matrix))
        self.wait(2)
        
        em_algorithm = Text("Learn Action-Conditional Transition Matrix", font_size=20)
        em_algorithm.move_to(DOWN * 2)
        self.play(Write(em_algorithm))
        self.wait(2)
        self.play(FadeOut(transition_matrix), FadeOut(em_algorithm))

    def create_viterbi_training(self):
        viterbi = Text("Viterbi Training for Parameter Re-estimation", font_size=20)
        viterbi.move_to(ORIGIN)
        self.play(Write(viterbi))
        self.wait(2)
        self.play(FadeOut(viterbi))

    def create_noisy_observations(self):
        noisy_obs = Text("Handling Noisy Observations", font_size=20)
        noisy_obs.move_to(UP * 2)
        self.play(Write(noisy_obs))
        self.wait(2)
        self.play(FadeOut(noisy_obs))

    def create_place_fields(self):
        place_fields = Text("Computing Place Fields in Test Environments", font_size=20)
        place_fields.move_to(DOWN * 2)
        self.play(Write(place_fields))
        self.wait(2)
        self.play(FadeOut(place_fields))

    def finalize_scene(self):
        final_text = Text("Optimization for Continuous State-Conditional GMMs", font_size=24)
        final_text.move_to(ORIGIN)
        self.play(Write(final_text))
        self.wait(2)
        self.play(FadeOut(final_text))

class HippocampalPlaceCellsAndSpatialCognitionScene(Scene):
    def construct(self):
        # Title
        title = Text("Hippocampal Place Cells and Spatial Cognition", font_size=36)
        self.play(Write(title))
        self.wait(1)

        # Description of hippocampal place cells
        description = Text("Activation of place cells related to position during trials.", font_size=24)
        description.next_to(title, DOWN, buff=0.5)
        self.play(Write(description))
        self.wait(2)

        # Create a visual representation of a maze (spatial navigation)
        maze = self.create_maze()
        self.play(FadeIn(maze))
        self.wait(2)

        # Animate place cells activation
        place_cells = self.create_place_cells(maze)
        self.play(ShowCreation(place_cells))
        self.wait(2)

        # Transition to discuss neuronal coding
        self.play(FadeOut(maze), FadeOut(place_cells))
        neuronal_coding = Text("Neuronal Coding and Spatial Cognition", font_size=30)
        self.play(Write(neuronal_coding))
        self.wait(1)

        # Illustrate neuronal activity
        neuron_activity = self.create_neuron_activity()
        self.play(FadeIn(neuron_activity))
        self.wait(2)

        # Comprehensive list of references
        references = Text("References on hippocampal function and spatial navigation.", font_size=24)
        references.next_to(neuronal_coding, DOWN, buff=0.5)
        self.play(Write(references))
        self.wait(2)

        # Final fade out
        self.play(FadeOut(title), FadeOut(description), FadeOut(neuronal_coding), FadeOut(references), FadeOut(neuron_activity))

    def create_maze(self):
        # Create a simple maze structure
        maze = VGroup()
        for i in range(4):
            maze.add(Line(LEFT * 2 + UP * i, RIGHT * 2 + UP * i).set_color(BLUE))
        for j in range(3):
            maze.add(Line(LEFT * 2 + UP * 1.5 + RIGHT * j, LEFT * 2 + UP * 1.5 + DOWN * 3).set_color(BLUE))
        return maze

    def create_place_cells(self, maze):
        # Create animated place cells
        place_cells = VGroup()
        for _ in range(5):
            cell = Dot(color=YELLOW).scale(0.5)
            cell.move_to(np.random.uniform(-2, 2, 2) + UP * 1.5)
            place_cells.add(cell)
        return place_cells

    def create_neuron_activity(self):
        # Create a representation of neuronal activity
        neuron_activity = VGroup()
        for _ in range(10):
            neuron = Dot(color=GREEN).scale(0.3)
            neuron.move_to(np.random.uniform(-3, 3, 2) + DOWN * 1)
            neuron_activity.add(neuron)
        return neuron_activity

class SpatialAndTemporalProcessingInTheHippocampusScene(Scene):
    def construct(self):
        self.create_title()
        self.create_spatial_firing_properties()
        self.create_sequence_generation()
        self.create_spatial_maps()
        self.create_place_and_grid_cells()
        self.create_final_transition()
        self.wait(2)

    def create_title(self):
        title = Text("Spatial and Temporal Processing in the Hippocampus", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

    def create_spatial_firing_properties(self):
        spatial_circle = Circle(radius=1, color=BLUE).shift(LEFT * 2)
        spatial_text = Text("Spatial Firing Properties", font_size=24).next_to(spatial_circle, DOWN)
        self.play(Create(spatial_circle), Write(spatial_text))
        self.wait(1)
        self.play(FadeOut(spatial_circle), FadeOut(spatial_text))

    def create_sequence_generation(self):
        sequence_line = Line(UP, DOWN, color=GREEN).shift(RIGHT * 2)
        sequence_text = Text("Sequence Generation", font_size=24).next_to(sequence_line, DOWN)
        self.play(Create(sequence_line), Write(sequence_text))
        self.wait(1)
        self.play(FadeOut(sequence_line), FadeOut(sequence_text))

    def create_spatial_maps(self):
        grid = NumberPlane()
        self.play(Create(grid))
        grid_label = Text("Multiple Spatial Maps", font_size=24).to_edge(UP)
        self.play(Write(grid_label))
        self.wait(1)
        self.play(FadeOut(grid), FadeOut(grid_label))

    def create_place_and_grid_cells(self):
        place_cell = Circle(radius=0.5, color=ORANGE).shift(LEFT * 2)
        grid_cell = Square(side_length=0.5, color=PURPLE).shift(RIGHT * 2)
        place_text = Text("Place Cells", font_size=24).next_to(place_cell, DOWN)
        grid_text = Text("Grid Cells", font_size=24).next_to(grid_cell, DOWN)
        
        self.play(Create(place_cell), Write(place_text))
        self.play(Create(grid_cell), Write(grid_text))
        self.wait(1)
        self.play(FadeOut(place_cell), FadeOut(place_text))
        self.play(FadeOut(grid_cell), FadeOut(grid_text))

    def create_final_transition(self):
        final_text = Text("Importance of Sensory Cues & Memory", font_size=28).scale(0.7).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)
        self.play(FadeOut(final_text))

