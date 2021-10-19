import { TestBed } from '@angular/core/testing';

import { NeuralnetworkService } from './neuralnetwork.service';

describe('NeuralnetworkService', () => {
  let service: NeuralnetworkService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(NeuralnetworkService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
