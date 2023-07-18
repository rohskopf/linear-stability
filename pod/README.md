### Condition of POD matrix

To test this, insert the following in `fitpod_command.cpp`, right above `// solving the linear system`:

    FILE * fh;
    fh = fopen("pod.dat", "w");
    for (int i = 0; i<nd; i++) {
      for (int j = 0; j<nd; j++){
        fprintf(fh, "%f ", desc.A[i + nd*j]);
      }
      fprintf(fh, "\n");
    }
    fclose(fh);

    // solving the linear system A * c = b

    int nrhs=1, info;
    char chu = 'U';
    DPOSV(&chu, &nd, &nrhs, desc.A, &nd, desc.c, &nd, &info);

Compile these changes and run

    lmp -in in.fitpod

Then calculate condition of the POD matrix with:

    python calc.py