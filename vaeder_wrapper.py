import json
import torch
from typing import Dict, List
from neutone_midi_sdk import MidiToMidiBase, NeutoneParameter, TokenData, prepare_token_data


class VaederModelWrapper(MidiToMidiBase):

    def get_model_name(self) -> str:
        return "vaeder"

    def get_model_authors(self) -> List[str]:
        return ["Julian Lenz"]

    def get_model_short_description(self) -> str:
        return "Tap2Drum"

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("density", "pattern density", default_value=0.65),
            NeutoneParameter("intensity", "pattern intensity", default_value=0.5),
            NeutoneParameter("rock", "rock", default_value=1.0),
            NeutoneParameter("jazz", "jazz", default_value=0.0)
        ]

    def get_hits_activation(self, _h: torch.Tensor, threshold: float=0.5):
        _h = torch.sigmoid(_h)
        h = torch.where(_h > threshold, 1, 0)
        return h

    def do_forward_pass(self, tokenized_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:


        n_patterns = tokenized_data.shape[0]

        density, intensity, rock, jazz = params["density"], \
            params["intensity"], params["rock"], params["jazz"]

        # The razz knob will swing between 'rock' and 'jazz' genre encodings at elements 0 and 2
        genre = torch.zeros((n_patterns, 16))
        genre[:, 0] = rock
        genre[:, 2] = jazz

        density = density.repeat(n_patterns)
        intensity = intensity.repeat(n_patterns)

        hvo, mu, log_var, z = self.model.forward(tokenized_data, density, intensity, genre)
        h, v, o = hvo

        # Sampling process
        hits = self.get_hits_activation(h, threshold=0.5)
        velocities = torch.sigmoid(v)
        offsets = torch.sigmoid(o) - 0.5
        hvo_tensor = torch.cat([hits, velocities, offsets], dim=-1)

        return hvo_tensor



if __name__ == "__main__":

    tokenizer_type = "HVO_collapsed"
    vaeder_model = torch.jit.load("earthy_149.pt")

    wrapped_model = VaederModelWrapper(model=vaeder_model,
                                       vocab=None,
                                       tokenizer_type=tokenizer_type,
                                       tokenizer_data=None,
                                       add_dimension=False)
    scripted_model = torch.jit.script(wrapped_model)


    scripted_model.save("earthy_nt.pt")

    # TEST
    input = torch.tensor([[0.0, 64.0, 62.0, 1.0]])
    params = torch.tensor([0.6, 0.45, 0.05, 0.5])
    output = scripted_model.forward(input, params)

    print(output.shape)

