import com.cedarsoftware.util.io.JsonWriter;
import com.gorbash.neural.network.BPNeuralNetwork;
import com.gorbash.neural.trainer.BPNNTrainer;
import com.gorbash.neural.trainer.NeuralNetworkMonitor;
import com.gorbash.neural.trainer.TrainingElement;
import com.gorbash.neural.trainer.TrainingManager;

import java.io.FileWriter;
import java.util.List;

import static java.util.Arrays.asList;

/**
 * Created by GorbasH on 3/9/2015.
 */
public class Functional {

    public static void main(String[] args) throws Exception {

        List<TrainingElement> suite = asList(
                new TrainingElement(asList(0.0, 0.0), asList(0.0, 0.0)),
                new TrainingElement(asList(0.0, 1.0), asList(1.0, 0.0)),
                new TrainingElement(asList(1.0, 0.0), asList(1.0, 0.0)),
                new TrainingElement(asList(1.0, 1.0), asList(0.0, 1.0)));

        BPNeuralNetwork network = new BPNeuralNetwork.BPPNeuralNetworkBuilder(2, 2).randomizeBias().setHiddenLayers(asList(2)).build();
        BPNNTrainer trainer = BPNNTrainer.build(network, 2.0);
        NeuralNetworkMonitor monitor = new NeuralNetworkMonitor(network);

        TrainingManager manager = new TrainingManager(trainer, 1000);
        double before = monitor.getTotalError(suite);
        manager.runTraining(suite);
        double after = monitor.getTotalError(suite);

        System.out.println(String.format("Before: %s, After: %s", before, after));


        FileWriter writer = new FileWriter("network.json");
        writer.write(JsonWriter.objectToJson(network));
        writer.close();


    }
}
